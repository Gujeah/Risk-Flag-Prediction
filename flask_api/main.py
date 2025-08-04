import mlflow
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import traceback
import logging

# Configure logging for the application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model Loading Logic ---
def load_registered_model(model_name: str, stage: str = None, version: str = None):
    """
    Loads a model from the MLflow Model Registry using its name and stage or version.
    """
    # Set the MLflow tracking URI (ensure this IP is current)
    mlflow.set_tracking_uri("http://ec2-54-198-215-161.compute-1.amazonaws.com:5000/")
    
    try:
        if stage:
            model_uri = f"models:/{model_name}/{stage}"
        elif version:
            model_uri = f"models:/{model_name}/{version}"
        else:
            raise ValueError("Either 'stage' or 'version' must be specified.")
            
        logger.info(f"Loading model from URI: {model_uri}")
        
        # Load the model using mlflow.pyfunc
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from MLflow Model Registry: {e}")
        # Log the full traceback for debugging purposes
        traceback.print_exc()
        return None

# Flask API Setup 
app = Flask(__name__)


MODEL_NAME = "lightgbm_risk_flag_model"
MODEL_VERSION = "1"  
loaded_model = load_registered_model(MODEL_NAME, version=MODEL_VERSION)

if loaded_model is None:
    logger.error("Failed to load model. API will not be able to make predictions.")

@app.route('/')
def home():
    """Simple welcome message for the home page."""
    return "Welcome to the Risk Flag Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predicts a risk flag based on input data.
    Input should be a JSON object with a list of dictionaries, one for each sample.
    """
    if loaded_model is None:
        return jsonify({'error': 'Model not loaded. Cannot make predictions.'}), 503
        
    if not request.json:
        return jsonify({'error': 'Invalid request. Please provide JSON data.'}), 400
    
    try:
        # Convert JSON input to a pandas DataFrame
        input_df = pd.DataFrame(request.json)
        logger.info(f"Received prediction request with data: {input_df}")
        
        # Make predictions
        predictions = loaded_model.predict(input_df)
        
        # Return predictions as a JSON response
        return jsonify({'predictions': predictions.tolist()})
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        traceback.print_exc()
        return jsonify({'error': 'An unexpected error occurred during prediction.'}), 500

if __name__ == "__main__":
    # The Flask development server will run on all network interfaces (0.0.0.0) on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)