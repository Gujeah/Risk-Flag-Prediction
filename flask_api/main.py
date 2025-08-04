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
        # Reverting to the generic pyfunc wrapper
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

if loaded_model is None or city_mapping is None or state_mapping is None or city_global_mean is None or state_global_mean is None:
    logger.error("Failed to load all necessary components. API will not make predictions.")
    
@app.route('/')
def home():
    """Renders the home page with the prediction form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predicts a risk flag based on raw user inputs."""
    if loaded_model is None or city_mapping is None or state_mapping is None or city_global_mean is None or state_global_mean is None:
        return render_template('index.html', prediction_message="Error: Components not loaded.", result="N/A")

    try:
        form_data = request.form.to_dict()

        required_fields = ['income', 'age', 'experience', 'current_job_yrs', 'current_house_yrs', 'city', 'state']
        for field in required_fields:
            if field not in form_data or not form_data[field]:
                return render_template('index.html', prediction_message=f"Input Error: Missing or empty field '{field}'.", result="N/A")

        try:
            raw_data = {
                'income': float(form_data['income']),
                'age': float(form_data['age']),
                'experience': float(form_data['experience']),
                'current_job_yrs': float(form_data['current_job_yrs']),
                'current_house_yrs': float(form_data['current_house_yrs']),
                'city': form_data['city'],
                'state': form_data['state'],
            }
        except ValueError as e:
            return render_template('index.html', prediction_message=f"Input Error: Invalid number format. Details: {e}", result="N/A")
        
        if raw_data['age'] <= 0:
            return render_template('index.html', prediction_message="Input Error: Age must be a positive number.", result="N/A")
        if raw_data['experience'] < 0:
            return render_template('index.html', prediction_message="Input Error: Experience cannot be negative.", result="N/A")

        user_df = pd.DataFrame([raw_data])
        user_df['income_to_age'] = user_df['income'] / user_df['age']
        user_df['income_to_experience'] = user_df['income'] / (user_df['experience'] + 1)
        user_df['stability_score'] = (user_df['current_job_yrs'] + user_df['current_house_yrs']) / user_df['age']

        user_df['city_encoded'] = user_df['city'].map(city_mapping).fillna(city_global_mean)
        user_df['state_encoded'] = user_df['state'].map(state_mapping).fillna(state_global_mean)
        
        final_features = ['city_encoded', 'state_encoded', 'income_to_age', 'income_to_experience', 'stability_score', 'experience']
        prediction_df = user_df[final_features].copy()
        prediction_df['experience'] = prediction_df['experience'].astype(int)

        logger.info(f"DataFrame to predict: {prediction_df}")

        # Reverting to simple binary prediction
        prediction = loaded_model.predict(prediction_df)[0]
        result_message = "RISK FLAG: YES (High Risk)" if prediction == 1 else "RISK FLAG: NO (Low Risk)"

        return render_template('index.html', prediction_message=result_message, result=prediction)

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        traceback.print_exc()
        return render_template('index.html', prediction_message="Internal Server Error during prediction.", result="N/A")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)