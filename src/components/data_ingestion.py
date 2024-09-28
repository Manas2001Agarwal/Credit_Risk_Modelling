import os
import sys
import warnings
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig
warnings.filterwarnings("ignore")

from sklearn import set_config
set_config(transform_output = "default")
pd.set_option("display.max_columns",None)

@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join("artifacts","train.csv")
    test_data_path : str = os.path.join("artifacts","test.csv")
    val_data_path : str = os.path.join("artifacts","val.csv")
    raw_data_path : str = os.path.join("artifacts","raw.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
        
    def initiate_data_ingestion(self):
        logging.info("enter the data ingestion/component")
        try:
            a1 = pd.read_excel("data/case_study1.xlsx")
            a2 = pd.read_excel("data/case_study2.xlsx")
            
            df1 = a1.copy()
            df2 = a2.copy()
            
            df1 = df1.loc[df1['Age_Oldest_TL'] != -99999]
            
            columns_to_be_removed = []
            for i in df2.columns:
                if df2.loc[df2[i] == -99999].shape[0] > 10000:
                    columns_to_be_removed .append(i)
                    
            df2 = df2.drop(columns_to_be_removed, axis =1)
            
            for i in df2.columns:
                df2 = df2.loc[ df2[i] != -99999 ]
                
            self.df = pd. merge ( df1, df2, how ='inner', left_on = ['PROSPECTID'], right_on = ['PROSPECTID'] )
            logging.info("read the data as dataframe, removed null rows and columns, and finally joined the two datasets")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            self.df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info("train-test-split-initiated")
            train_set,test_set = train_test_split(self.df,test_size=0.2,random_state = 42)
            train_set,val_set = train_test_split(train_set,test_size=0.2,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            val_set.to_csv(self.ingestion_config.val_data_path,index=False,header=True)
            
            logging.info("Ingestion of the data iss completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.val_data_path
            )
        except Exception as e:
            raise CustomException(e,sys) # type: ignore
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data,val_data = obj.initiate_data_ingestion()
    
    data_transformation  = DataTransformation()
    train_arr,val_arr,_ = data_transformation.initiate_data_transformation(train_data,val_data)
    
    model_train = ModelTrainer()
    print(model_train.initiate_model_training(train_arr,val_arr))
    