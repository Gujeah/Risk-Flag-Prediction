# src/modeling/model_evaluation.py

import numpy as np
import pandas as pd
import joblib
import logging
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from mlflow.models import infer_signature

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler('model_evaluation.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.info('Data loaded from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise


def load_model(model_path: str):
    """Load the trained model (using joblib)."""
    try:
        model = joblib.load(model_path)
        logger.info('Model loaded from %s', model_path)
        return model
    except Exception as e:
        logger.error('Error loading model from %s: %s', model_path, e)
        raise


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.info('Parameters loaded from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading parameters from %s: %s', params_path, e)
        raise


def evaluate_model_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray):
    """Calculate and return a dictionary of evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc_score': roc_auc_score(y_true, y_prob)
    }
    logger.info('Model evaluation metrics calculated.')
    return metrics

# New function to save metrics to a local file
def save_metrics(metrics: dict, path: str) -> None:
    """Saves evaluation metrics to a JSON file for DVC."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Convert numpy types to native Python types for JSON serialization
        metrics_serializable = {k: float(v) if isinstance(v, np.generic) else v for k, v in metrics.items()}
        with open(path, 'w') as f:
            json.dump(metrics_serializable, f, indent=4)
        logger.info('Metrics saved to %s for DVC tracking.', path)
    except Exception as e:
        logger.error('Error occurred while saving metrics for DVC: %s', e)
        raise

def log_confusion_matrix(cm, dataset_name, plot_path):
    """Log confusion matrix as an artifact."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    plt.close()
    logger.info(f'Confusion matrix plot saved to {plot_path} and logged to MLflow.')


def save_model_info(run_id: str, model_artifact_path: str, file_path: str) -> None:
    """Save the MLflow run ID and model artifact path to a JSON file."""
    try:
        model_info = {
            'run_id': run_id,
            'model_artifact_path': model_artifact_path
        }
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.info('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise


def main():
    mlflow.set_tracking_uri("http://ec2-54-198-215-161.compute-1.amazonaws.com:5000/")
    mlflow.set_experiment('Risk-Flag-Prediction-Evaluation-V1')

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    
    params_path = os.path.join(root_dir, 'params.yaml')
    train_data_path = os.path.join(root_dir, 'data/processed/train_processed.csv')
    model_path = os.path.join(root_dir, 'models/lightgbm_model.joblib')
    
    metrics_json_path = os.path.join(root_dir, 'metrics/validation_metrics.json')
    confusion_matrix_plot_path = os.path.join(root_dir, 'plots/confusion_matrix_validation.png')
    model_info_json_path = os.path.join(root_dir, 'mlflow_run_info.json')

    with mlflow.start_run() as run:
        try:
            logger.info(f"MLflow Run ID: {run.info.run_id}")

            params = load_params(params_path)
            
            mlflow.log_param("random_state", params.get('random_state'))
            if 'model' in params and 'lightgbm' in params['model']:
                for param_name, param_value in params['model']['lightgbm'].items():
                    mlflow.log_param(f"lgbm_{param_name}", param_value)

            model = load_model(model_path)

            train_data = load_data(train_data_path)

            X = train_data.drop('risk_flag', axis=1)
            y = train_data['risk_flag']

            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=params.get('random_state', 42))
            train_idx, val_idx = next(kf.split(X, y))
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            input_example = X_val.head(5)
            output_example = model.predict(input_example)
            signature = infer_signature(input_example, output_example)

            mlflow.sklearn.log_model(
                model,
                "lightgbm_risk_model",
                signature=signature,
                input_example=input_example
            )
            logger.info("Model logged to MLflow with signature and input example.")

            save_model_info(run.info.run_id, "lightgbm_risk_model", model_info_json_path)

            y_pred_val = model.predict(X_val)
            y_prob_val = model.predict_proba(X_val)[:, 1]
            
            metrics = evaluate_model_metrics(y_val, y_pred_val, y_prob_val)
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"validation_{metric_name}", metric_value)
            logger.info(f"Validation Metrics logged to MLflow: {metrics}")

            # New: Save metrics to a local JSON file for DVC
            save_metrics(metrics, metrics_json_path)

            cm = confusion_matrix(y_val, y_pred_val)
            log_confusion_matrix(cm, "Validation Data", confusion_matrix_plot_path)

            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Risk Flag Prediction")
            mlflow.set_tag("dataset", "Customer Data")
            mlflow.set_tag("evaluation_type", "Validation Set")
            logger.info("MLflow tags set.")

        except Exception as e:
            logger.error(f"Failed to complete model evaluation: {e}")
            print(f"Error: {e}")
            mlflow.set_tag("status", "Failed")
            raise

if __name__ == '__main__':
    main()