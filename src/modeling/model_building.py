# src/modeling/model_building.py

import logging
import os
import joblib
import pandas as pd
import yaml
from lightgbm import LGBMClassifier

# --- Logging configuration ---
logger = logging.getLogger('model_building')
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
file_handler = logging.FileHandler('model_building.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_name: str, path: str) -> pd.DataFrame:
    """Loads a processed dataset from the specified path."""
    file_path = os.path.join(path, file_name)
    try:
        df = pd.read_csv(file_path)
        logger.info('Data loaded from %s', file_path)
        return df
    except FileNotFoundError:
        logger.error('File not found at %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error happened when loading the dataset: %s', e)
        raise

if __name__ == '__main__':
    # --- DVC input/output paths ---
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
    model_save_path = os.path.join(base_path, 'models/lightgbm_model.joblib')
    params_path = os.path.join(base_path, 'params.yaml')

    try:
        # Load parameters from params.yaml
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        
        # Load the processed training data
        df_train = load_data('train_processed.csv', os.path.join(base_path, 'data/processed/'))

        # Separate features and target
        X = df_train.drop('risk_flag', axis=1)
        y = df_train['risk_flag']

        # Get the LightGBM parameters from the nested dictionary
        lgbm_params = params['model']['lightgbm']

        # Initialize and train the LightGBM model on the full training dataset
        model = LGBMClassifier(**lgbm_params)
        model.fit(X, y)
        
        logger.info('Model trained on the full training dataset.')
        
        # Save the final model
        joblib.dump(model, model_save_path)
        logger.info('Model saved to %s', model_save_path)
        logger.info("Final model building pipeline completed successfully.")

    except Exception as e:
        logger.critical('Final model building pipeline failed with a critical error: "%s"', e)
        print(f"Error: {e}")
        exit(1)