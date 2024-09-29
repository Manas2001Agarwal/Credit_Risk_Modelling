import sys
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
import warnings
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    OrdinalEncoder,
    FunctionTransformer,
    LabelEncoder
)
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.components.custom_transformer import One_way_annova, VIF

sklearn.set_config(transform_output="default")

@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    target_transformer_obj_file_path = os.path.join('artifacts','target_trf.pkl')
    
class DataTransformation:
    '''
    This class is used to transform data
    '''
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def trans_edu(self,X):
        return (
            X.replace(['SSC','12TH','GRADUATE',
                       'UNDER GRADUATE','POST-GRADUATE',
                       'OTHERS','PROFESSIONAL'],
                        
                        [1,2,3,3,4,1,3])
        )
        
    def get_data_transformer_object(self,X):
        try:
            categorical_columns = ['MARITALSTATUS', 'GENDER' , 'last_prod_enq2' ,'first_prod_enq2']
            
            numeric_columns = []
            for i in X.columns:
                if X[i].dtype != 'object' and i not in ['PROSPECTID','Approved_Flag']:
                    numeric_columns.append(i)
                    
            cat_pipeline = Pipeline(steps=[
                ('ohe',OneHotEncoder(sparse_output=False)),
            ])
            
            cat_edu = FunctionTransformer(self.trans_edu)
            
            num_pipeline = Pipeline(steps=[
                        ('vif',VIF(6)),
                        ('annova',One_way_annova()) ])
            
            logging.info("Categorical columns encoding completed")
            
            preprocessor = ColumnTransformer(transformers = [
                ('num_pipeline',num_pipeline,numeric_columns),
                ('cat_pipe',cat_pipeline,categorical_columns),
                ('cat_edu',cat_edu,['EDUCATION'])
            ])
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys) # type: ignore
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_data = pd.read_csv("artifacts/train.csv")
            test_data = pd.read_csv("artifacts/test.csv")
            
            logging.info("Read_train and test data completed")
            preprocessing_obj = self.get_data_transformer_object(train_data)
            target_column_name = "Approved_Flag"
            lb = LabelEncoder()
            
            input_feature_train_df = train_data.drop(columns = [target_column_name],axis=1)
            target_feature_train_df=lb.fit_transform(train_data[target_column_name])
            
            input_feature_test_df=test_data.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=lb.transform(test_data[target_column_name])

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df,train_data[target_column_name])
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            save_object(
                file_path=self.data_transformation_config.target_transformer_obj_file_path,
                obj = lb
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys) # type: ignore
        