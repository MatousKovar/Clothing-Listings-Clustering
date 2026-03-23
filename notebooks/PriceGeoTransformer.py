import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class PriceGeoTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, exchange_rates=None):
        self.exchange_rates = exchange_rates or {}
    def fit(self, X, y=None):        
        X['price'] = X['price'].astype(float)
        
        price_eur = X.apply(lambda row: self.convert_price(row['geo'], row['price']), axis=1)
        self.geo_means_ = price_eur.groupby(X['geo']).mean().to_dict()
        self.geo_stds_ = price_eur.groupby(X['geo']).std().replace(0, 1.0).to_dict()
        
        return self
        


    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=['price', 'geo'])
            
        X['price'] = X['price'].astype(float)
        
        price_eur = X.apply(lambda row: self.convert_price(row['geo'], row['price']), axis=1)
        
        means = X['geo'].map(self.geo_means_).fillna(self.global_mean_)
        stds = X['geo'].map(self.geo_stds_).fillna(self.global_std_)
        
        price_scaled_by_geo = (price_eur - means) / stds
        
        return np.c_[price_eur, price_scaled_by_geo]
    
    def convert_price(geo, price):
        if geo == "cz":
            return price / 24.5
        elif geo == "hu":
            return price / 388.5
        elif geo == "ro":
            return price / 5.09
        elif geo == "pl":
            return price / 4.7
        else:
            return price