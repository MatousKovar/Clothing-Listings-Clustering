import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DepartmentIdsCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=['departmentIds'])
        else:
            X = X.copy()
            
        col = 'departmentIds'
        X[col] = X[col].fillna("").astype(str)
        X[col] = X[col].str.replace(r"\[|\]|'|\s", "", regex=True)
        
        return X[[col]]
    
    def get_feature_names_out(self, input_features=None):
        return np.array(['departmentIds'])