# data_preprocessing.py

import numpy as np
import pandas as pd
import logging
import os 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import joblib 
import pickle

# --- Logging configuration ---
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
file_handler = logging.FileHandler('preprocessing.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_raw_data(file_name: str, path: str) -> pd.DataFrame:
    """Loads a raw dataset from the specified path."""
    file_path = os.path.join(path, file_name)
    try:
        df = pd.read_csv(file_path)
        logger.info('Raw data loaded from %s', file_path)
        return df
    except FileNotFoundError:
        logger.error('File not found at %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error happened when loading the dataset: %s', e)
        raise

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses a single dataframe with consistent steps."""
    try:
        df = df.copy() 
        df.drop_duplicates(inplace=True)
        if 'id' in df.columns:
            df.set_index("id", inplace=True)
        df.columns = df.columns.str.lower().str.replace(" ", "_")

        if 'house_ownership' in df.columns:
            df['house_ownership'] = df['house_ownership'].replace('norent_noown', 'norent_known')

        for col in ['city', 'state', 'profession']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(r'\[.*?\]','', regex=True)
                df[col] = df[col].astype(str).str.lower().str.replace(" ", "_")

        if 'profession' in df.columns:
            conditions = [
                df['profession'].isin(['physician', 'surgeon', 'dentist', 'psychologist', 'biomedical_engineer']),
                df['profession'].isin(['mechanical_engineer', 'chemical_engineer', 'industrial_engineer', 'civil_engineer', 'design_engineer', 'computer_hardware_engineer', 'petroleum_engineer']),
                df['profession'].isin(['software_developer', 'web_designer', 'computer_operator', 'technology_specialist']),
                df['profession'].isin(['artist', 'designer', 'fashion_designer', 'graphic_designer']),
                df['profession'].isin(['chef', 'firefighter', 'hotel_manager', 'flight_attendant']),
                df['profession'].isin(['secretary', 'librarian', 'technical_writer', 'official', 'statistician', 'drafter', 'computer_operator']),
                df['profession'].isin(['lawyer', 'magistrate', 'chartered_accountant', 'financial_analyst']),
                df['profession'].isin(['police_officer', 'army_officer', 'air_traffic_controller'])
            ]
            choices = [
                'healthcare', 'engineering', 'technology', 'arts/design', 'services', 'office/admin', 'legal', 'public_safety'
            ]
            df['profession_grouped'] = np.select(conditions, choices, default='other')
            df.drop(columns="profession", inplace=True)

        if 'income' in df.columns and 'experience' in df.columns:
            df['income_to_experience'] = df['income'] / (df['experience'] + 1)
        if 'current_job_yrs' in df.columns and 'current_house_yrs' in df.columns and 'age' in df.columns:
            df['stability_score'] = (df['current_job_yrs'] + df['current_house_yrs']) / df['age']
        if 'income' in df.columns and 'age' in df.columns:
            df['income_to_age'] = df['income'] / df['age']

        logger.info('Data preprocessing completed.')
        return df
    except KeyError as e:
        logger.error('Missing column during preprocessing: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame to a specified path."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logger.info('Data saved to %s', path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

# --- FIX: Modified the function to return all 4 values ---
def cross_validated_target_encode(df_train: pd.DataFrame, y_train: pd.Series, df_test: pd.DataFrame, col: str, n_splits: int = 5) -> tuple:
    """Performs cross-validated target encoding for a given column."""
    oof_train = pd.Series(np.nan, index=df_train.index)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for tr_idx, val_idx in kf.split(df_train, y_train):
        X_tr, X_val = df_train.iloc[tr_idx], df_train.iloc[val_idx]
        y_tr = y_train.iloc[tr_idx]
        
        # Calculate target mean for each category in the training fold
        target_means = y_tr.groupby(X_tr[col]).mean()
        
        # Map the target means to the validation fold
        oof_train.iloc[val_idx] = X_val[col].map(target_means)

    # Calculate the final target means on the full training data
    final_target_means = y_train.groupby(df_train[col]).mean()
    
    # Map the final target means to the test data
    oof_test = df_test[col].map(final_target_means)
    
    # Fill any NaNs (new categories in test data) with the global mean
    global_mean = y_train.mean()
    oof_test = oof_test.fillna(global_mean)
    
    return oof_train, oof_test, final_target_means, global_mean # Now returns 4 values

if __name__ == '__main__':
    # --- DVC input/output paths ---
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
    dvc_raw_path = os.path.join(base_path, 'data/raw')
    dvc_processed_path = os.path.join(base_path, 'data/processed')

    try:
        # Load raw data from the DVC raw directory
        df_train_raw = load_raw_data('train.csv', dvc_raw_path)
        df_test_raw = load_raw_data('test.csv', dvc_raw_path)

        # Preprocess both datasets
        df_train_preprocessed = preprocess(df_train_raw)
        df_test_preprocessed = preprocess(df_test_raw)
        
        # --- Split features and target for training data ---
        X_train_full = df_train_preprocessed.drop('risk_flag', axis=1)
        y_train_full = df_train_preprocessed['risk_flag']
        
        # --- Test data has no target variable ---
        X_test_final = df_test_preprocessed.copy()
        
        # --- FIX: Changed the function call to capture all 4 values ---
        oof_train_city, oof_test_city, city_mapping, city_global_mean = cross_validated_target_encode(
            df_train=X_train_full, y_train=y_train_full, df_test=X_test_final, col='city'
        )
        X_train_full['city_encoded'] = oof_train_city
        X_test_final['city_encoded'] = oof_test_city

        oof_train_state, oof_test_state, state_mapping, state_global_mean = cross_validated_target_encode(
            df_train=X_train_full, y_train=y_train_full, df_test=X_test_final, col='state'
        )
        X_train_full['state_encoded'] = oof_train_state
        X_test_final['state_encoded'] = oof_test_state

        # saving artifacts
        artifacts_path = os.path.join(base_path, 'artifacts')
        os.makedirs(artifacts_path, exist_ok=True)
        
        joblib.dump(city_mapping, os.path.join(artifacts_path, 'city_mapping.joblib'))
        joblib.dump(city_global_mean, os.path.join(artifacts_path, 'city_global_mean.joblib'))
        joblib.dump(state_mapping, os.path.join(artifacts_path, 'state_mapping.joblib'))
        joblib.dump(state_global_mean, os.path.join(artifacts_path, 'state_global_mean.joblib'))
        logger.info("Target encoding artifacts saved.")
        
        # Perform Label Encoding on low-cardinality columns
        le_cols = ['married/single', 'house_ownership', 'car_ownership', 'profession_grouped']
        for col in le_cols:
            le = LabelEncoder()
            if col in X_train_full.columns:
                X_train_full[col] = le.fit_transform(X_train_full[col])
                if col in X_test_final.columns:
                    X_test_final[col] = le.transform(X_test_final[col])

        # Final feature selection
        top_features = ['city_encoded', 'state_encoded', 'income_to_age', 'income_to_experience', 'stability_score', 'experience']
        X_train_final = X_train_full[top_features]
        X_test_final = X_test_final[top_features]

        # Save the processed data
        train_data = pd.concat([X_train_final, y_train_full], axis=1)
        test_data = X_test_final
        
        save_data(train_data, os.path.join(dvc_processed_path, "train_processed.csv"))
        save_data(test_data, os.path.join(dvc_processed_path, "test_processed.csv"))
        
        logger.info("Preprocessing pipeline completed successfully.")

    except Exception as e:
        logger.critical('Preprocessing pipeline failed with a critical error: "%s"', e)
        print(f"Error: {e}")
        exit(1)