
import pandas as pd
import numpy as np
import logging
import os 
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder

#  Logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('preprocessing_errors.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_raw_data(file_name: str, path: str) -> pd.DataFrame:
    """Loads a raw dataset from the specified path."""
    file_path = os.path.join(path, 'raw', file_name)
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

def save_processed_data(df: pd.DataFrame, file_name: str, path: str) -> None:
    """Saves a processed DataFrame to the specified path."""
    try:
        processed_data_path = os.path.join(path, 'processed')
        os.makedirs(processed_data_path, exist_ok=True)
        df.to_csv(os.path.join(processed_data_path, file_name), index=False)
        logger.info('Processed data saved to %s', os.path.join(processed_data_path, file_name))
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    """Main function for the data preprocessing pipeline."""
    try:
        # Define the DVC raw and processed data paths
        data_base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data')

        # Load raw data from the DVC raw directory
        df_train_raw = load_raw_data('train.csv', data_base_path)
        df_test_raw = load_raw_data('test.csv', data_base_path)

        # Preprocess both datasets
        df_train_preprocessed = preprocess(df_train_raw)
        df_test_preprocessed = preprocess(df_test_raw)
        
        # --- Split features and target for encoders ---
        X_train_orig = df_train_preprocessed.drop('risk_flag', axis=1)
        y_train = df_train_preprocessed['risk_flag']
        X_test_orig = df_test_preprocessed.drop('risk_flag', axis=1)
        y_test = df_test_preprocessed['risk_flag']

        # --- Correct Target Encoding without Data Leakage ---
        target_encoder_city = TargetEncoder(smoothing=10)
        target_encoder_state = TargetEncoder(smoothing=10)
        
        target_encoder_city.fit(X_train_orig['city'], y_train)
        target_encoder_state.fit(X_train_orig['state'], y_train)
        
        X_train_orig['city_encoded'] = target_encoder_city.transform(X_train_orig['city'])
        X_test_orig['city_encoded'] = target_encoder_city.transform(X_test_orig['city'])
        X_train_orig['state_encoded'] = target_encoder_state.transform(X_train_orig['state'])
        X_test_orig['state_encoded'] = target_encoder_state.transform(X_test_orig['state'])
        
        le_cols = ['married/single', 'house_ownership', 'car_ownership', 'profession_grouped']
        for col in le_cols:
            le = LabelEncoder()
            if col in X_train_orig.columns:
                X_train_orig[col] = le.fit_transform(X_train_orig[col])
                X_test_orig[col] = le.transform(X_test_orig[col])

        top_features = ['city_encoded', 'state_encoded', 'income_to_age', 'income_to_experience', 'stability_score', 'experience']
        X_train_final = X_train_orig[top_features]
        X_test_final = X_test_orig[top_features]

        # Save the processed data
        train_data = pd.concat([X_train_final, y_train], axis=1)
        test_data = pd.concat([X_test_final, y_test], axis=1)
        
        save_processed_data(train_data, "train_processed.csv", data_base_path)
        save_processed_data(test_data, "test_processed.csv", data_base_path)
        
        logger.info("Preprocessing pipeline completed successfully.")

    except Exception as e:
        logger.critical('Preprocessing pipeline failed with a critical error: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()