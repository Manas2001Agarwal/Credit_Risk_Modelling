import pandas as pd
import numpy as np
import dill
from sklearn.pipeline import Pipeline
from src.components.custom_transformer import VIF, One_way_annova
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
import streamlit as st # type: ignore

preprocessor_obj = "/Users/mukulagarwal/Desktop/Python_Code/Classification_Project/artifacts/preprocessor.pkl"
with open(preprocessor_obj,"rb") as file_p:
    preprocessor = dill.load(file_p)

model_obj = "/Users/mukulagarwal/Desktop/Python_Code/Classification_Project/artifacts/model.pkl"
with open(model_obj,"rb") as file_m:
    model = dill.load(file_m)
    
target_trf_path = "/Users/mukulagarwal/Desktop/Python_Code/Classification_Project/artifacts/target_trf.pkl"
with open(target_trf_path,"rb") as file_t:
    target_trf = dill.load(file_t)
    
        
st.write('''
         ## Input Data
         ''')
uploaded_file = st.file_uploader("Choose a file")

def user_input_features():
    if uploaded_file is not None:
            dataframe = pd.read_csv(uploaded_file)
            st.write(dataframe.iloc[:,:-1])
    return dataframe

input_df = user_input_features()


if st.button('Run Model'):
    test_X = preprocessor.transform(input_df.iloc[:,:-1])
    prediction = model.predict(test_X)
    transformed_target = target_trf.transform(input_df.iloc[:,-1])
    st.metric('Accuracy',round(accuracy_score(transformed_target,prediction)*100,2))
    
    p_score = precision_score(transformed_target,prediction,average='macro')
    st.metric('Precision Score',round(p_score,4)) #type: ignore
    
    r_score = recall_score(transformed_target,prediction,average='macro')
    st.metric('Recall Score',round(r_score,4)) #type: ignore
