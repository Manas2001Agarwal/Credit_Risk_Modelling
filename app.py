import pandas as pd
import numpy as np
import dill
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
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

# Function to plot confusion matrix
def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=['Predicted 0', 'Predicted 1','Predicted 2','Predicted 3'],
                yticklabels=['True 0', 'True 1','True 2','True 3'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

if st.button('Run Model'):
    test_X = preprocessor.transform(input_df.iloc[:,:-1])
    prediction = model.predict(test_X)
    transformed_target = target_trf.transform(input_df.iloc[:,-1])
    
    col1, col2 = st.columns([0.5, 2])  # Adjust the ratios as needed

    with col1:
        st.metric('Accuracy',round(accuracy_score(transformed_target,prediction)*100,2))
        
        p_score = precision_score(transformed_target,prediction,average='macro')
        st.metric('Precision Score',round(p_score,4)) #type: ignore
        
        r_score = recall_score(transformed_target,prediction,average='macro')
        st.metric('Recall Score',round(r_score,4)) #type: ignore
        
        predict_proba = model.predict_proba(test_X)
        roc_auc_score_ = roc_auc_score(transformed_target,predict_proba,multi_class='ovr')
        st.metric('roc_auc_score',round(roc_auc_score_,4)) #type: ignore
        
        f_score = f1_score(transformed_target,prediction,average='macro')
        st.metric('f1_score',round(f_score,4)) #type: ignore

    with col2:
        cm = confusion_matrix(transformed_target,prediction)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        plot_confusion_matrix(cm_normalized)
