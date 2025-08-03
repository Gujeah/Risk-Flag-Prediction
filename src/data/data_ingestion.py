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
        ### Changing column name formats
        df.columns=df.columns.str.lower().str.replace(" ", "_")
        #### For categorical columns
        cols=['married/single','house_ownership','car_ownership','profession','city','state']
        #changing wrong name "norent_noown" to "norent_known"
        df['house_ownership'] = df['house_ownership'].replace('norent_noown', 'norent_known')
        df.city=df.city.str.replace(r'\[.*?\]','',regex=True)

 







    except Exception as e:
