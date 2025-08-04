
import os
from sklearn.model_selection import train_test_split
import yaml
import logging
import pandas as pd
import json


#configurations
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('ingestion_errors.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_path: str) -> pd.DataFrame:
    """Loads a raw dataset from a specified path."""
    try:
        if data_path.endswith('.json'):
            df = pd.read_json(data_path)
        elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
            df = pd.read_excel(data_path)
        else:
            raise ValueError("Unsupported file format. Please use .csv or .xlsx")
            
        logger.info('Loaded data from %s', data_path)
        return df
    except FileNotFoundError:
        logger.error('File not found at %s', data_path)
        raise
    except Exception as e:
        logger.error('Unexpected error happened when loading the dataset %s: %s', data_path, e)
        raise

def save_raw_data(df: pd.DataFrame, file_name: str, path: str) -> None:
    """Saves a DataFrame to the raw data directory."""
    try:
        raw_data_path = os.path.join(path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        df.to_csv(os.path.join(raw_data_path, file_name))
        logger.info('Raw data saved to %s', os.path.join(raw_data_path, file_name))
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    """Main function for the data ingestion pipeline."""
    try:
        # data source
        train_source_path = "https://raw.githubusercontent.com/Isaac-Jim/ZionTech_Hackathon/refs/heads/main/train.json"
        test_source_path = "https://raw.githubusercontent.com/Isaac-Jim/ZionTech_Hackathon/refs/heads/main/test.json"
        
        # project structure
        data_destination_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data')

        # Loading  and save the training data
        df_train = load_data(train_source_path)
        save_raw_data(df_train, 'train.csv', data_destination_path)

        # Loading and saving the test data
        df_test = load_data(test_source_path)
        save_raw_data(df_test, 'test.csv', data_destination_path)
        
        logger.info("Raw data ingestion pipeline completed successfully.")

    except Exception as e:
        logger.critical('Ingestion pipeline failed: %s', e)

if __name__ == '__main__':
    main()