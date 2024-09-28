from sklearn.base import TransformerMixin, BaseEstimator
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import f_oneway
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

from sklearn import set_config
set_config(transform_output = "default")

class VIF(BaseEstimator,TransformerMixin):
    
    def __init__(self,vif_threshold):
        self.vif_threshold = vif_threshold
    
    def fit(self,X,y):
        numeric_columns = []
        for i in X.columns:
            if X[i].dtype != 'object' and i not in ['PROSPECTID','Approved_Flag']:
                numeric_columns.append(i)
                
        self.vif_data = X[numeric_columns]
        total_columns = self.vif_data.shape[1]
        self.columns_to_be_kept = []
        column_index = 0
        
        for i in range (0,total_columns):
            vif_value = variance_inflation_factor(self.vif_data, column_index)
            if vif_value <= 6:
                self.columns_to_be_kept.append( numeric_columns[i] )
                column_index = column_index+1
            else:
                self.vif_data = self.vif_data.drop([ numeric_columns[i] ],axis=1)
        return self
    
    def transform(self,X):
       return X[self.columns_to_be_kept]
   
class One_way_annova(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.columns_to_be_kept = []
    
    def fit(self,X,y):
        total_columns = X.columns
        for i in total_columns:
                a = list(X[i])  
                b = list(y)  
                
                group_P1 = [value for value, group in zip(a, b) if group == 'P1']
                group_P2 = [value for value, group in zip(a, b) if group == 'P2']
                group_P3 = [value for value, group in zip(a, b) if group == 'P3']
                group_P4 = [value for value, group in zip(a, b) if group == 'P4']


                f_statistic, p_value = f_oneway(group_P1, group_P2, group_P3, group_P4)

                if p_value <= 0.05:
                    self.columns_to_be_kept.append(i)
        return self
    
    def transform(self,X):
        return np.array(X[self.columns_to_be_kept])
    
fs_pipe = Pipeline(steps=[
    ('vif',VIF(6)),
    ('annova',One_way_annova())
])

if __name__ == "__main__":
    raw_data = pd.read_csv("artifacts/train.csv")
    X = raw_data.iloc[:,:-1]
    y = raw_data.iloc[:,-1]
    print(fs_pipe.fit_transform(X,y))
   