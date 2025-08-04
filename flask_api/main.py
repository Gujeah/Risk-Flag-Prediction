import mlflow
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import traceback
import logging
import os

# Configure logging for the application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Preprocessing Artifact Paths ---
# Use the artifacts directory from the project root
base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
artifacts_path = os.path.join(base_path, 'artifacts')

CITY_MAPPING_PATH = os.path.join(artifacts_path, "city_mapping.joblib")
STATE_MAPPING_PATH = os.path.join(artifacts_path, "state_mapping.joblib")
CITY_GLOBAL_MEAN_PATH = os.path.join(artifacts_path, "city_global_mean.joblib")
STATE_GLOBAL_MEAN_PATH = os.path.join(artifacts_path, "state_global_mean.joblib")

# --- Model Loading Logic ---
def load_registered_model(model_name: str, version: str = "1"):
    """Loads a model from the MLflow Model Registry."""
    # NOTE: Update the IP address here to your current EC2 instance's Public IPv4
    mlflow.set_tracking_uri("http://ec2-54-198-215-161.compute-1.amazonaws.com:5000/")
    
    try:
        model_uri = f"models:/{model_name}/{version}"
        logger.info(f"Loading model from URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from MLflow: {e}")
        traceback.print_exc()
        return None

# --- Flask API Setup ---
app = Flask(__name__)

# Load model and preprocessing artifacts once at application startup
MODEL_NAME = "lightgbm_risk_flag_model"
MODEL_VERSION = "1"
loaded_model = load_registered_model(MODEL_NAME, version=MODEL_VERSION)

try:
    city_mapping = joblib.load(CITY_MAPPING_PATH)
    state_mapping = joblib.load(STATE_MAPPING_PATH)
    city_global_mean = joblib.load(CITY_GLOBAL_MEAN_PATH)
    state_global_mean = joblib.load(STATE_GLOBAL_MEAN_PATH)
    logger.info("Preprocessing artifacts loaded successfully.")
except FileNotFoundError as e:
    city_mapping, state_mapping = None, None
    city_global_mean, state_global_mean = None, None
    logger.error(f"Failed to load artifacts: {e}")

if not all([loaded_model, city_mapping, state_mapping, city_global_mean, state_global_mean]):
    logger.error("Failed to load all necessary components. API will not make predictions.")

@app.route('/')
def home():
    """Renders the home page with the prediction form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predicts a risk flag based on raw user inputs."""
    if not all([loaded_model, city_mapping, state_mapping, city_global_mean, state_global_mean]):
        return render_template('index.html', prediction_message="Error: Components not loaded.", result="N/A")

    try:
        # 1. Get raw data from the form submission
        # We now expect 'current_job_yrs' and 'current_house_yrs' instead of 'stability_score'
        raw_data = {
            'income': float(request.form['income']),
            'age': float(request.form['age']),
            'experience': float(request.form['experience']),
            'current_job_yrs': float(request.form['current_job_yrs']),
            'current_house_yrs': float(request.form['current_house_yrs']),
            'city': request.form['city'],
            'state': request.form['state'],
        }

        # 2. Perform feature engineering and preprocessing
        user_df = pd.DataFrame([raw_data])
        user_df['income_to_age'] = user_df['income'] / user_df['age']
        user_df['income_to_experience'] = user_df['income'] / user_df['experience']
        
        # --- FIX: Calculate stability_score from raw inputs ---
        user_df['stability_score'] = (user_df['current_job_yrs'] + user_df['current_house_yrs']) / user_df['age']

        # Apply the loaded mappings and handle unseen categories with the global mean
        user_df['city_encoded'] = user_df['city'].map(city_mapping).fillna(city_global_mean)
        user_df['state_encoded'] = user_df['state'].map(state_mapping).fillna(state_global_mean)
        
        # 3. Select the final features for the model
        final_features = ['city_encoded', 'state_encoded', 'income_to_age', 'income_to_experience', 'stability_score', 'experience']
        prediction_df = user_df[final_features]

        # 4. Make prediction
        prediction = loaded_model.predict(prediction_df)[0]
        result_message = "RISK FLAG: YES (High Risk)" if prediction == 1 else "RISK FLAG: NO (Low Risk)"

        return render_template('index.html', prediction_message=result_message, result=prediction)

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        traceback.print_exc()
        return render_template('index.html', prediction_message="Error in prediction process", result="N/A")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)