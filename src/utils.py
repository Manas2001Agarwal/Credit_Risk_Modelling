import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import accuracy_score,classification_report, precision_recall_fscore_support,f1_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys) # type: ignore
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        print("model_training_started")
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            print(model)
            para=param[list(models.keys())[i]]
            print(para)
            gs = GridSearchCV(model,para,cv=3,scoring='accuracy',n_jobs=-1)
            
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y_train, y_train_pred)
            print("train_accuracy",train_model_score)

            test_model_score = accuracy_score(y_test, y_test_pred)
            print("test_accuracy",test_model_score)
            print()

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys) # type: ignore
        
