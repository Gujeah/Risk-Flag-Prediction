# src/modeling/model_evaluation.py

import json
import logging
import os
import joblib
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# --- Logging configuration ---
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
file_handler = logging.FileHandler('model_evaluation.log')
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

def evaluate_model(y_true, y_pred, y_prob):
    """Calculates and returns a dictionary of evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc_score': roc_auc_score(y_true, y_prob)
    }
    return metrics

def save_metrics(metrics: dict, path: str) -> None:
    """Saves evaluation metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info('Metrics saved to %s', path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving metrics: %s', e)
        raise

def save_confusion_matrix(y_true, y_pred, path: str) -> None:
    """Creates and saves a confusion matrix plot."""
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix on Validation Data')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        logger.info('Confusion matrix plot saved to %s', path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the plot: %s', e)
        raise

if __name__ == '__main__':
    # --- DVC input/output paths ---
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
    model_path = os.path.join(base_path, 'models/lightgbm_model.joblib')
    processed_data_path = os.path.join(base_path, 'data/processed/')
    metrics_path = os.path.join(base_path, 'metrics/validation_metrics.json')
    plots_path = os.path.join(base_path, 'plots/confusion_matrix_validation.png')

    try:
        # Load parameters from params.yaml
        with open(os.path.join(base_path, 'params.yaml'), 'r') as f:
            params = yaml.safe_load(f)

        # Load the processed training data
        df_train = load_data('train_processed.csv', processed_data_path)

        # Separate features from target
        X = df_train.drop('risk_flag', axis=1)
        y = df_train['risk_flag']

        # Perform a validation split
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=params['random_state'])
        train_idx, val_idx = next(kf.split(X, y))
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # Load the trained model
        model = load_model(model_path)

        # Make predictions on the validation data
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        # Evaluate the model
        metrics = evaluate_model(y_val, y_pred, y_prob)
        logger.info('Validation Metrics: %s', metrics)

        # Save metrics to JSON
        save_metrics(metrics, metrics_path)

        # Save confusion matrix plot
        save_confusion_matrix(y_val, y_pred, plots_path)

        logger.info("Model evaluation pipeline completed successfully.")

    except Exception as e:
        logger.critical('Model evaluation pipeline failed with a critical error: "%s"', e)
        print(f"Error: {e}")
        exit(1)