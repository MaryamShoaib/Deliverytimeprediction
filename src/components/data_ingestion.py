import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation


## Initialize the data ingestion

@dataclass
class DataIngestioncofig:
    train_data_path:str=os.path.join('artifacts', 'train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestioncofig()

    def split_column(self, df, column_name):  #function to split date column and change data type
        try:
            df['Order_Day'] = df[column_name].str.split('-').str[0]
            df['Order_Month'] = df[column_name].str.split('-').str[1]
            df['Order_Year'] = df[column_name].str.split('-').str[2]

            df.drop(column_name, axis=1, inplace=True)

            df['Order_Day'] = df['Order_Day'].astype(int)
            df['Order_Month'] = df['Order_Month'].astype(int)
            df['Order_Year'] = df['Order_Year'].astype(int)

            logging.info('Date split done')
            return df

        except Exception as e:
            logging.info("Exception occurred in the initiate_datatransformation")
            raise CustomException(e, sys)

    def split_time(self, df, column_name): #function to split time and change data type
        def format_time(time_str):
            if isinstance(time_str, str) and '.' in time_str:
                decimal_time = float(time_str)
                hours = int(decimal_time)
                minutes = round((decimal_time - hours) * 60)
                return f"{hours:02d}:{minutes:02d}"
            else:
                return time_str
        try:
            df[column_name] = df[column_name].apply(format_time) #changing decimal values into hours and mins
            df[column_name+'Hour'] = df[column_name].str.split(':').str[0]  #spliting hours and mins
            df[column_name+'Min'] = df[column_name].str.split(':').str[1]
            df.drop(column_name, axis=1, inplace=True)

            #Change nan into 0
            df[column_name+'Min'] = df[column_name+'Min'].fillna(0)
            df[column_name+'Hour'] = df[column_name+'Hour'].fillna(0)

            #Changing dtype to int
            df[column_name+'Hour'] = df[column_name+'Hour'].astype(int)
            df[column_name+'Min'] = df[column_name+'Min'].astype(int)
            logging.info('Split time done')

            return df
        except Exception as e:
            logging.info("Exception occurred at split time")
            raise CustomException(e, sys)

    
    def initiate_data_ingestion(self):
        logging.info('Data Ingestion has started')
        try:
            
            df=pd.read_csv(os.path.join('notebooks/data','finalTrain.csv'))
            logging.info('Data read as pandas Dataframe')

            df=self.split_time(df,'Time_Orderd')
            df=self.split_time(df,'Time_Order_picked')
            df=self.split_column(df,'Order_Date')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info('Train test split')
            train_set, test_set=train_test_split(df,test_size=0.30, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Ingestion of Data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )

        except Exception as e:
            logging.info('Exception occured at Data Ingestion')
            raise CustomException (e,sys)
        


## run Data Ingestion

if __name__=='__main__':
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()