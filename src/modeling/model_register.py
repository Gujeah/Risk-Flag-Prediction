# src/modeling/model_registration.py

import json
import mlflow
import logging
import os

# Set up MLflow tracking URI
mlflow.set_tracking_uri("http://ec2-54-198-215-161.compute-1.amazonaws.com:5000/")

# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.info('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise


def register_model(model_name: str, model_info: dict) -> None:
    """Registers a model to the MLflow Model Registry."""
    try:
        run_id = model_info.get('run_id')
        model_artifact_path = model_info.get('model_artifact_path')
        
        if not run_id or not model_artifact_path:
            raise ValueError("Model info JSON is missing 'run_id' or 'model_artifact_path'")

        model_uri = f"runs:/{run_id}/{model_artifact_path}"
        
        # Register the model
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        logger.info(f"Model '{model_name}' successfully registered to the MLflow Model Registry.")
        logger.info(f"Version: {registered_model.version}")
        logger.info(f"Name: {registered_model.name}")
        
    except Exception as e:
        logger.error(f'Error occurred during model registration: {e}')
        raise

def main():
    try:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        
        # The new path for the MLflow run info file
        model_info_path = os.path.join(root_dir, 'mlflow_run_info.json') 
        model_info = load_model_info(model_info_path)
        
        # A descriptive name for the model in the registry
        model_name = "lightgbm_risk_flag_model"
        
        register_model(model_name, model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()