import numpy as np
import pandas as pd
import logging
import os 
from sklearn.model_selection import train_test_split
import yaml
## loging congiguration
logger=logging.getLogger('data_ingestion')
logger.set_level(logging.DEBUG)
##console handler
console_handler=logging.StreamHandler()
console_handler.set_level(logging.DEBUG)

##file handling 
file_handler=logging.FileHandler('errors.log')
file_handler.set_level(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

## A function to load of dattaset 
def load_data(data_url: str) -> pd.DataFrame:
    try:
        df=pd.read_csv(data_url)
        logger.debug('loaded data from %s', data_url)
        return df
    except pd.errors.ParserError as e: 
        logger.error('unexpected error happend when loading the dataset %s', data_url)
        raise
# we should now preprocess our dataframe 
def preprocess(df:pd.DataFrame) -> pd.DataFrame:
    try:
        ## dropping duplicates
        df.drop_duplicates(inplace=True)
        ## setting an index
        df.set_index("id", inplace=True)
        ### Changing column name formats
        df.columns=df.columns.str.lower().str.replace(" ", "_")
        #### For categorical columns
        cols=['married/single','house_ownership','car_ownership','profession','city','state']
        #changing wrong name "norent_noown" to "norent_known"
        df['house_ownership'] = df['house_ownership'].replace('norent_noown', 'norent_known')
        ### removing the charactors trailing spaces and the likes 
        df.city=df.city.str.replace(r'\[.*?\]','',regex=True)
        df.state=df.state.str.replace(r'\[.*?\]','',regex=True)
        ### Changing categories to lower ans adding _ to spaces for bettter use
        categorical_columns=list(df.dtypes[df.dtypes=="object"].index)
        for c in categorical_columns:
            df[c]=df[c].str.lower().str.replace(" ", "_")
        #### Grouping everything on professional columns since it has high dimension
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
        df['profession_grouped'] = np.select(conditions, choices, default='Other')
        df.drop(columns="profession", inplace=True)
        ###category encoding for those columns with very high dimensionality
        X_train_full = df.drop('risk_flag', axis=1)
        y_train_full= df['risk_flag']
        # Initialize columns
        X_train_full['city_encoded'] = np.nan
        X_train_full['state_encoded'] = np.nan
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X_train_full, y_train_full):
            X_tr, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
            y_tr = y_train_full.iloc[train_idx]
        ### encoding for city and states 
        # City encoding
        city_encoder = TargetEncoder(smoothing=10)
        city_encoder.fit(X_tr['city'], y_tr)
        X_train_full.loc[X_train_full.index[val_idx], 'city_encoded'] = \
            city_encoder.transform(X_val['city']).values.ravel()
        # State encoding
        state_encoder = TargetEncoder(smoothing=10)
        state_encoder.fit(X_tr['state'], y_tr)
        X_train_full.loc[X_train_full.index[val_idx], 'state_encoded'] = \
            state_encoder.transform(X_val['state']).values.ravel()
        # Fit final encoders on full data to use on test set
        final_city_encoder = TargetEncoder(smoothing=10).fit(X_train_full['city'], y_train_full)
        final_state_encoder = TargetEncoder(smoothing=10).fit(X_train_full['state'], y_train_full)
        df['city_encoded'] = final_city_encoder.transform(df['city']).values.ravel()
        df['state_encoded'] = final_state_encoder.transform(df['state']).values.ravel()
        ###concatinating it back to normal dataframe 
        df=pd.concat([df,  y_train_full], axis=1)
        ### Perfoming label encoding on categorical columns
        low_card_cols = ['married/single', 'house_ownership', 'car_ownership', 'profession_grouped']
        for col in low_card_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        df['income_to_experience'] =df['income'] / (df['experience'] + 1)
        df['stability_score'] = (df['current_job_yrs'] + df['current_house_yrs']) / df['age']
        df['income_to_age'] = df['income'] / df['age']
        ## KEeeping these features for model training: top_5_features = ['city_encoded', 'income_to_age', 'income_to_experience', 'stability_score', 'experience']
        ##features we will use in our model
        df=df['city_encoded', 'income_to_age', 'income_to_experience', 'stability_score', 'experience']
        logger.debug('Data preprocessing completed: So many things were happening wheew!!!')
        return df

    except KeyError as e:
        logger.error('missing columns %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets, creating the raw folder if it doesn't exist."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        
        # Create the data/raw directory if it does not exist
        os.makedirs(raw_data_path, exist_ok=True)
        
        # Save the train and test data
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    try:
        df=load_data(link)
        final_df=preprocess(df)
        save_data(train_data, test_data, data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data'))
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")
if __name__=='main':
    main()
