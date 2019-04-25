# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 12:24:24 2019

@author: dgill
"""

import os
from dirutil import project_directory
from configfile import get_config

import pandas as pd, numpy as np

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


class StandardLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelEncoder(*args, **kwargs)
    def fit(self, X, y=None):
        self.encoder.fit(X)
        return self
    def transform(self, X):
        return self.encoder.transform(X)


class MultiColumnLabelEncoder:
    def __init__(self, *args, **kwargs):
        self.encoder = StandardLabelEncoder(*args, **kwargs)
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        data = X.copy()
        for i in range(data.shape[1]):
            data[:, i] = LabelEncoder().fit_transform(data[:, i])
        return data
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class ReshapeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, shape):
        self.shape = shape
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.reshape(self.shape)


class DataTypeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.astype(self.dtype)


if __name__ == '__main__':
    config = get_config()
    
    target = config['target']
    categorical_predictors = config['cat_predictors']
    numerical_predictors = config['num_predictors']
    
    diw_path = os.path.join(
        project_directory(), config['diw_path'], 'diw.txt'        
    )
    
    diw_df = pd.read_csv(
        diw_path, sep=config['data_sep']        
    )
    
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(numerical_predictors)),
        ('imputer', Imputer()),
        ('scaler', MinMaxScaler())               
    ])
    # The cat vars we have now don't require imputing
    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(categorical_predictors)),
        ('label_encoder', MultiColumnLabelEncoder()),
        ('one_hot_encoder', OneHotEncoder(sparse=False))
    ])
    target_pipeline = Pipeline([
        ('selector', DataFrameSelector(target)),
        ('dtype_transform', DataTypeTransformer(np.int8)),
        ('reshape', ReshapeTransformer((-1, 1))),
        ('one_hot_encoder', OneHotEncoder(sparse=False))        
    ])
    
    join_pipeline = FeatureUnion(transformer_list=[
        ('num_pipeline', num_pipeline),
        ('cat_pipeline', cat_pipeline),
        ('target_pipeline', target_pipeline)
    ])
    
    d = join_pipeline.fit_transform(diw_df)
    model_path = os.path.join(
        project_directory(), config['diw_path'], 'model_data.csv'        
    )
    np.savetxt(model_path, d, delimiter='|')
    
    
    
    
    
    