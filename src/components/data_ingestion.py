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

pd.set_option("display.max_columns",None)

@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join("artifacts","train.csv")
    test_data_path : str = os.path.join("artifacts","test.csv")
    raw_data_path : str = os.path.join("artifacts","raw.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def vif_num(self):
        numeric_columns = []
        for i in self.df.columns:
            if self.df[i].dtype != 'object' and i not in ['PROSPECTID','Approved_Flag']:
                numeric_columns.append(i)
            
        vif_data = self.df[numeric_columns]
        total_columns = vif_data.shape[1]
        columns_to_be_kept = []
        column_index = 0
                 
        for i in range (0,total_columns):
    
            vif_value = variance_inflation_factor(vif_data, column_index)
            if vif_value <= 6:
                columns_to_be_kept.append( numeric_columns[i] )
                column_index = column_index+1
                
            else:
                vif_data = vif_data.drop([ numeric_columns[i] ] , axis=1)
        columns_to_be_kept_numerical = []

        for i in columns_to_be_kept:
            a = list(self.df[i])  
            b = list(self.df['Approved_Flag'])  
            
            group_P1 = [value for value, group in zip(a, b) if group == 'P1']
            group_P2 = [value for value, group in zip(a, b) if group == 'P2']
            group_P3 = [value for value, group in zip(a, b) if group == 'P3']
            group_P4 = [value for value, group in zip(a, b) if group == 'P4']


            f_statistic, p_value = f_oneway(group_P1, group_P2, group_P3, group_P4)

            if p_value <= 0.05:
                columns_to_be_kept_numerical.append(i)
        return columns_to_be_kept_numerical

        
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
            
            # VIF for numerical columns
            columns_to_be_kept_numerical = self.vif_num()
            features = columns_to_be_kept_numerical + ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
            self.df = self.df[features + ['Approved_Flag']]

            
            logging.info("train-test-split-initiated")
            train_set,test_set = train_test_split(self.df,test_size=0.2,random_state = 42)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("Ingestion of the data iss completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys) # type: ignore
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()
    
    data_transformation  = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data,test_data)
    # print(train_arr)
    # print()
    # print(test_arr)
    # print()
    
    model_train = ModelTrainer()
    print(model_train.initiate_model_training(train_arr,test_arr))
    