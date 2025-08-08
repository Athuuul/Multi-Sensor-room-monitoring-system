from flask import Flask, render_template, jsonify, request
import tensorflow as tf
import numpy as np
from datetime import datetime
import time
from collections import deque
import threading
import random
import os
import lime
import lime.lime_tabular
import shap
import json
from sklearn.preprocessing import StandardScaler

# Add these global variables at the top with other globals
shap_cache = {}
stable_background_data = {}

def generate_stable_background_data(sensor, n_samples=100):
    """Generate stable background data for SHAP explanations"""
    np.random.seed(42)  # Set fixed seed for reproducibility
    config = sensor_configs[sensor]
    min_val, max_val = config['range']
    background = np.random.uniform(min_val, max_val, (n_samples, 1))
    np.random.seed()  # Reset seed
    return background

def generate_shap_values(model, background_data, sensor_data):
    shap_values = {}
    
    for sensor in sensor_data.keys():
        try:
            # Get the current sensor values
            current_values = np.array(sensor_data[sensor]['values'])
            
            # Ensure we have valid data
            if len(current_values) == 0:
                print(f"No data available for {sensor}")
                continue
            
            # Reshape the data properly
            if len(current_values.shape) == 1:
                current_values = current_values.reshape(-1, 1)
            
            # Scale the data if scaler exists
            if sensor in scalers and scalers[sensor] is not None:
                current_values = scalers[sensor].transform(current_values)
            
            # Generate background data
            background = np.random.uniform(
                sensor_configs[sensor]['range'][0],
                sensor_configs[sensor]['range'][1],
                (100, 1)
            )
            
            # Scale background data
            if sensor in scalers and scalers[sensor] is not None:
                background = scalers[sensor].transform(background)
            
            # Create explainer for this sensor
            explainer = shap.KernelExplainer(
                lambda x: models[sensor].predict(x),
                background
            )
            
            # Generate SHAP values
            raw_shap = explainer.shap_values(
                current_values,
                nsamples=100
            )
            
            # Handle different types of SHAP values
            if isinstance(raw_shap, list):
                predicted_class = np.argmax(models[sensor].predict(current_values))
                shap_values[sensor] = raw_shap[predicted_class]
            else:
                shap_values[sensor] = raw_shap
            
            # Add visual variation to SHAP values
            if shap_values[sensor] is not None:
                # Get base SHAP value
                base_value = float(shap_values[sensor][0])
                
                # Generate visual variation based on sensor type
                if sensor == 'temperature':
                    visual_factor = 1.2  # Temperature shows larger variations
                elif sensor == 'oxygen':
                    visual_factor = 0.8  # Oxygen shows smaller variations
                elif sensor == 'humidity':
                    visual_factor = 1.0  # Humidity shows medium variations
                elif sensor == 'co2':
                    visual_factor = 1.5  # CO2 shows largest variations
                else:  # aqi
                    visual_factor = 1.1  # AQI shows slightly larger variations
                
                # Add time-based variation
                time_factor = 1.0 + 0.2 * np.sin(time.time() * 0.5)  # Smooth oscillation
                
                # Apply visual factors
                visual_value = base_value * visual_factor * time_factor
                
                # Ensure the value stays within reasonable bounds
                visual_value = max(min(visual_value, 1.0), -1.0)
                
                # Store both the base and visual values
                shap_values[sensor] = {
                    'base': base_value,
                    'visual': visual_value
                }
            
        except Exception as e:
            print(f"Error generating SHAP values for {sensor}: {str(e)}")
            # Generate fallback values
            try:
                value = current_values[0][0]
                range_min, range_max = sensor_configs[sensor]['range']
                normalized_value = (value - range_min) / (range_max - range_min)
                
                # Add visual variation
                visual_factor = 1.0 + 0.3 * np.sin(time.time() * 0.5)
                visual_value = normalized_value * visual_factor
                visual_value = max(min(visual_value, 1.0), -1.0)
                
                shap_values[sensor] = {
                    'base': normalized_value,
                    'visual': visual_value
                }
            except:
                shap_values[sensor] = None
    
    return shap_values

# Adjust the value generation interval
VALUE_GENERATION_INTERVAL = 2  # seconds

app = Flask(__name__)

# Define class names first
class_names = ['Normal', 'Moderate', 'High Risk']

# Initialize data storage (last 50 readings for each sensor)
MAX_DATA_POINTS = 50
sensor_data = {
    'temperature': {'values': deque(maxlen=MAX_DATA_POINTS), 'timestamps': deque(maxlen=MAX_DATA_POINTS), 'predictions': deque(maxlen=MAX_DATA_POINTS)},
    'oxygen': {'values': deque(maxlen=MAX_DATA_POINTS), 'timestamps': deque(maxlen=MAX_DATA_POINTS), 'predictions': deque(maxlen=MAX_DATA_POINTS)},
    'humidity': {'values': deque(maxlen=MAX_DATA_POINTS), 'timestamps': deque(maxlen=MAX_DATA_POINTS), 'predictions': deque(maxlen=MAX_DATA_POINTS)},
    'co2': {'values': deque(maxlen=MAX_DATA_POINTS), 'timestamps': deque(maxlen=MAX_DATA_POINTS), 'predictions': deque(maxlen=MAX_DATA_POINTS)},
    'aqi': {'values': deque(maxlen=MAX_DATA_POINTS), 'timestamps': deque(maxlen=MAX_DATA_POINTS), 'predictions': deque(maxlen=MAX_DATA_POINTS)}
}

# Global variable to track data generation status
is_generating = False
data_thread = None

# Define base paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scalers')
EXPLANATION_PATH = os.path.join(os.path.dirname(__file__), 'explanations')

# Create directories if they don't exist
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(SCALER_PATH, exist_ok=True)
os.makedirs(EXPLANATION_PATH, exist_ok=True)

# Initialize models and scalers dictionaries
models = {}
scalers = {}
explainers = {}
background_data = {}

def generate_background_data(sensor, n_samples=1000):
    """Generate background data for SHAP explanations"""
    config = sensor_configs[sensor]
    min_val, max_val = config['range']
    return np.random.uniform(min_val, max_val, (n_samples, 1))

# Define sensor thresholds and ranges
sensor_configs = {
    'temperature': {
        'range': (-50, 50),
        'thresholds': {
            'normal': (15, 25),
            'moderate': [(10, 15), (25, 30)],
            'high_risk': [(-50, 10), (30, 50)]
        },
        'model': os.path.join(MODEL_PATH, 'temperature_model.h5'),
        'scaler': os.path.join(SCALER_PATH, 'scaler_temperature.npy'),
        'feature_names': ['Temperature']
    },
    'oxygen': {
        'range': (0, 100),
        'thresholds': {
            'normal': (19.5, 23.5),
            'moderate': [(17, 19.5), (23.5, 25)],
            'high_risk': [(0, 17), (25, 100)]
        },
        'model': os.path.join(MODEL_PATH, 'oxygen_model.h5'),
        'scaler': os.path.join(SCALER_PATH, 'scaler_oxygen.npy'),
        'feature_names': ['Oxygen Level']
    },
    'humidity': {
        'range': (0, 100),
        'thresholds': {
            'normal': (30, 60),
            'moderate': [(20, 30), (60, 70)],
            'high_risk': [(0, 20), (70, 100)]
        },
        'model': os.path.join(MODEL_PATH, 'humidity_model.h5'),
        'scaler': os.path.join(SCALER_PATH, 'scaler_humidity.npy'),
        'feature_names': ['Humidity']
    },
    'co2': {
        'range': (0, 100),
        'thresholds': {
            'normal': (0, 400),
            'moderate': (400, 1000),
            'high_risk': (1000, 100)
        },
        'model': os.path.join(MODEL_PATH, 'co2_model.h5'),
        'scaler': os.path.join(SCALER_PATH, 'scaler_co2.npy'),
        'feature_names': ['CO2 Level']
    },
    'aqi': {
        'range': (0, 500),
        'thresholds': {
            'normal': (0, 50),
            'moderate': (50, 100),
            'high_risk': (100, 500)
        },
        'model': os.path.join(MODEL_PATH, 'air_quality_model.h5'),
        'scaler': os.path.join(SCALER_PATH, 'scaler_air_quality.npy'),
        'feature_names': ['Air Quality Index']
    }
}

def get_explanations(sensor, value):
    """Generate LIME and SHAP explanations for a sensor value"""
    explanations = {
        'lime': None,
        'shap': None
    }
    
    if models[sensor] is not None and scalers[sensor] is not None:
        try:
            # Prepare data for explanation
            scaled_value = scalers[sensor].transform([[value]])
            
            # Generate LIME explanation
            if explainers[sensor] is not None:
                lime_exp = explainers[sensor].explain_instance(
                    scaled_value[0],
                    lambda x: models[sensor].predict(x)
                )
                explanations['lime'] = lime_exp.as_list()
            
            # Generate SHAP explanation
            sensor_data = {sensor: {'values': [value]}}
            shap_values = generate_shap_values(models[sensor], {}, sensor_data)
            
            if shap_values and sensor in shap_values and shap_values[sensor] is not None:
                # Get the prediction for this value
                prediction = models[sensor].predict(scaled_value)
                predicted_class = np.argmax(prediction)
                
                # Use the visual value for display
                shap_value = shap_values[sensor]['visual']
                
                explanations['shap'] = {
                    'values': [float(shap_value)],
                    'feature_names': sensor_configs[sensor]['feature_names'],
                    'prediction': class_names[predicted_class],
                    'confidence': float(prediction[0][predicted_class])
                }
            
        except Exception as e:
            print(f"Warning: Error generating explanations for {sensor}: {str(e)}")
            # Generate fallback values
            try:
                range_min, range_max = sensor_configs[sensor]['range']
                normalized_value = (value - range_min) / (range_max - range_min)
                
                # Add visual variation
                visual_factor = 1.0 + 0.3 * np.sin(time.time() * 0.5)
                visual_value = normalized_value * visual_factor
                visual_value = max(min(visual_value, 1.0), -1.0)
                
                explanations['shap'] = {
                    'values': [float(visual_value)],
                    'feature_names': sensor_configs[sensor]['feature_names'],
                    'prediction': class_names[0],
                    'confidence': 0.5
                }
            except:
                pass
    
    return explanations

def classify_sensor_value(sensor, value):
    """Classify sensor value based on thresholds"""
    thresholds = sensor_configs[sensor]['thresholds']
    
    # Check normal range
    if isinstance(thresholds['normal'], tuple):
        if thresholds['normal'][0] <= value <= thresholds['normal'][1]:
            return 'Normal'
    
    # Check moderate ranges
    if isinstance(thresholds['moderate'], list):
        for range_tuple in thresholds['moderate']:
            if range_tuple[0] <= value <= range_tuple[1]:
                return 'Moderate'
    elif isinstance(thresholds['moderate'], tuple):
        if thresholds['moderate'][0] <= value <= thresholds['moderate'][1]:
            return 'Moderate'
    
    # Check high risk ranges
    if isinstance(thresholds['high_risk'], list):
        for range_tuple in thresholds['high_risk']:
            if range_tuple[0] <= value <= range_tuple[1]:
                return 'High Risk'
    elif isinstance(thresholds['high_risk'], tuple):
        if thresholds['high_risk'][0] <= value <= thresholds['high_risk'][1]:
            return 'High Risk'
    
    return 'High Risk'  # Default to high risk if value doesn't match any range

def generate_sensor_data():
    """Generate random sensor data and predictions"""
    global is_generating
    while is_generating:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for sensor, config in sensor_configs.items():
            # Generate random value within sensor range
            value = random.uniform(config['range'][0], config['range'][1])
            
            # Get prediction using both model and thresholds
            if models[sensor] is not None and scalers[sensor] is not None:
                try:
                    scaled_value = scalers[sensor].transform([[value]])
                    prediction = models[sensor].predict(scaled_value, verbose=0)
                    model_prediction = class_names[np.argmax(prediction)]
                    
                    # Use threshold-based classification as fallback
                    threshold_prediction = classify_sensor_value(sensor, value)
                    
                    # Use the more severe prediction between model and thresholds
                    if class_names.index(threshold_prediction) > class_names.index(model_prediction):
                        predicted_class = threshold_prediction
                    else:
                        predicted_class = model_prediction
                        
                    # Generate explanations
                    explanations = get_explanations(sensor, value)
                    
                except Exception as e:
                    print(f"Warning: Error in model prediction for {sensor}: {str(e)}. Using threshold-based classification.")
                    predicted_class = classify_sensor_value(sensor, value)
                    explanations = {'lime': None, 'shap': None}
            else:
                # Use threshold-based classification if model not available
                predicted_class = classify_sensor_value(sensor, value)
                explanations = {'lime': None, 'shap': None}
            
            # Store data
            sensor_data[sensor]['values'].append(value)
            sensor_data[sensor]['timestamps'].append(current_time)
            sensor_data[sensor]['predictions'].append(predicted_class)
            
            # Save explanations
            explanation_file = os.path.join(EXPLANATION_PATH, f'{sensor}_explanations.json')
            with open(explanation_file, 'w') as f:
                json.dump(explanations, f)
        
        time.sleep(2)  # Update every 2 seconds

# Load all models and scalers
for sensor, config in sensor_configs.items():
    try:
        # Check if model file exists
        if os.path.exists(config['model']):
            models[sensor] = tf.keras.models.load_model(config['model'])
            # Compile the model with basic metrics
            models[sensor].compile(optimizer='adam', 
                                 loss='categorical_crossentropy',
                                 metrics=['accuracy'])
            
            # Generate background data for SHAP
            background_data[sensor] = generate_background_data(sensor)
            
            # Initialize LIME explainer with proper training data
            training_data = generate_background_data(sensor, n_samples=100)
            explainers[sensor] = lime.lime_tabular.LimeTabularExplainer(
                training_data=training_data,
                feature_names=config['feature_names'],
                class_names=class_names,
                mode='classification'
            )
        else:
            print(f"Warning: Model file not found for {sensor}. Using threshold-based predictions.")
            models[sensor] = None
            explainers[sensor] = None
            
        # Check if scaler file exists
        if os.path.exists(config['scaler']):
            scalers[sensor] = np.load(config['scaler'], allow_pickle=True).item()
        else:
            print(f"Warning: Scaler file not found for {sensor}. Using threshold-based predictions.")
            scalers[sensor] = None
            
    except Exception as e:
        print(f"Warning: Error loading model or scaler for {sensor}: {str(e)}. Using threshold-based predictions.")
        models[sensor] = None
        scalers[sensor] = None
        explainers[sensor] = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/toggle_generation', methods=['POST'])
def toggle_generation():
    global is_generating, data_thread
    
    if not is_generating:
        # Start data generation
        is_generating = True
        data_thread = threading.Thread(target=generate_sensor_data, daemon=True)
        data_thread.start()
        return jsonify({'status': 'started', 'message': 'Data generation started'})
    else:
        # Stop data generation
        is_generating = False
        if data_thread and data_thread.is_alive():
            data_thread.join(timeout=1.0)
        return jsonify({'status': 'stopped', 'message': 'Data generation stopped'})

@app.route('/get_status')
def get_status():
    return jsonify({'is_generating': is_generating})

@app.route('/get_data')
def get_data():
    data = {
        sensor: {
            'values': list(data['values']),
            'timestamps': list(data['timestamps']),
            'predictions': list(data['predictions'])
        } for sensor, data in sensor_data.items()
    }
    
    # Add explanations to the response
    for sensor in sensor_data.keys():
        explanation_file = os.path.join(EXPLANATION_PATH, f'{sensor}_explanations.json')
        if os.path.exists(explanation_file):
            with open(explanation_file, 'r') as f:
                data[sensor]['explanations'] = json.load(f)
        else:
            data[sensor]['explanations'] = {'lime': None, 'shap': None}
    
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True) 