import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score,classification_report, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
#from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig
        
    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("splitting training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                # "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                # #"XGBRegressor": XGBRegressor(),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                # "AdaBoost Regressor": AdaBoostClassifier(),
            }
            
            params={
                # "Decision Tree": {
                #     'splitter':['best','random'],
                #     'max_features':['sqrt','log2'],
                # },
                "Random Forest":{
                    'max_features':['sqrt','log2'],
                    'n_estimators': [320,420]
                },
                "Gradient Boosting":{
                    'subsample':[0.85,0.9],
                    'n_estimators': [320,420]
                },
                 "CatBoosting Classifier":{
                     'depth': [6,8,10],   
                 }
                #     "AdaBoost Classifier":{
                #     'learning_rate':[.1,.01,0.5,.001],
                #     'n_estimators': [8,16,32,64,128,256]
                # }
            }
            model_report : dict = evaluate_models(X_train = X_train,y_train = y_train, 
                                                  X_test = X_test, y_test = y_test, models = models,param=params)
            
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model found",sys) # type: ignore
            logging.info(f"Best found model on both training and testing dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            return accuracy
            

        except Exception as e:
            raise CustomException(e,sys) # type: ignore