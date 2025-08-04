# src/modeling/model_prediction.py

import logging
import os
import joblib
import pandas as pd
import numpy as np

# --- Logging configuration ---
logger = logging.getLogger('model_prediction')
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
file_handler = logging.FileHandler('model_prediction.log')
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

def load_model(path: str):
    """Loads the trained model from the specified path."""
    try:
        model = joblib.load(path)
        logger.info('Model loaded from %s', path)
        return model
    except FileNotFoundError:
        logger.error('Model not found at %s', path)
        raise
    except Exception as e:
        logger.error('Unexpected error happened when loading the model: %s', e)
        raise

def save_predictions(predictions: pd.DataFrame, path: str) -> None:
    """Saves predictions to a CSV file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        predictions.to_csv(path, index=False)
        logger.info('Predictions saved to %s', path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving predictions: %s', e)
        raise

if __name__ == '__main__':
    # --- DVC input/output paths ---
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
    model_path = os.path.join(base_path, 'models/lightgbm_model.joblib')
    processed_data_path = os.path.join(base_path, 'data/processed/')
    predictions_path = os.path.join(base_path, 'predictions/test_predictions.csv')

    try:
        # Load the processed test data (features only)
        df_test = load_data('test_processed.csv', processed_data_path)
        X_test = df_test.copy()

        # Load the trained model
        model = load_model(model_path)

        # Make predictions
        test_predictions = model.predict(X_test)
        
        # Save predictions to a DataFrame
        predictions_df = pd.DataFrame(test_predictions, columns=['risk_flag_prediction'])

        # Save predictions to CSV
        save_predictions(predictions_df, predictions_path)

        logger.info("Model prediction pipeline completed successfully.")

    except Exception as e:
        logger.critical('Model prediction pipeline failed with a critical error: "%s"', e)
        print(f"Error: {e}")
        exit(1)