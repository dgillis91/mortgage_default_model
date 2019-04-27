# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 19:41:06 2019

@author: dgill
"""

# !!!
# Switch to one hot and add more variables. 

# Had pretty good luck with KNN - may circle back around to this.
import os

from keras.models import Model
from keras.layers import Dense, BatchNormalization, Input

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder

from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import pandas as pd

from dirutil import project_directory
from configfile import get_config


def clean_nulls(df, cols):
    for col in cols:
        df[col].fillna((df[col].mean()), inplace=True)


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


class SamplerFactory:
    __sampler = {
        'over': SMOTE,
        'under': RandomUnderSampler
    }

    @staticmethod
    def get_instance(sample_method, *args, **kwargs):
        sampler = SamplerFactory.__sampler.get(sample_method, default=None)
        if sampler is not None:
            return sampler(*args, **kwargs)
        else:
            raise ValueError('invalid parameter: {}'.format(sample_method))


def false_positive(true_values, predicted_values):
    acc_df = pd.DataFrame(
        data=np.column_stack((true_values, predicted_values)),
        columns=['true_val', 'pred']
    )
    total_samples = len(acc_df.index)
    positives = len(acc_df[(acc_df.true_val == 0) & (acc_df.pred == 1)].index)
    return positives / total_samples


def false_negative(true_values, predicted_values):
    acc_df = pd.DataFrame(
        data=np.column_stack((true_values, predicted_values)),
        columns=['true_val', 'pred']        
    )
    total_samples = len(acc_df.index)
    negatives = len(acc_df[(acc_df.true_val == 1) & (acc_df.pred == 0)].index)
    return negatives / total_samples


if __name__ == '__main__':
    sample_method = 'under'
    
    project_path = project_directory()
    config = get_config()
    # todo: refactor all this crap
    train_pct = .9
    performance_data_path = os.path.join(
        project_path, config['diw_path'], 'diw.txt'
    )
    
    performance_data = pd.read_csv(performance_data_path, 
        sep=config['data_sep']
    )
    
    target = config['target']
    predictors = config['predictors']
    
    # Pull out the predictors & target
    model_data = performance_data[predictors + [target]]
    # del performance_data
    
    # Apply Transforms

    # Clean nulls and map f/c stat to bits
    clean_nulls(model_data, predictors)
    mapping = {True: 1, False: 0}
    model_data[target] = model_data[target].map(mapping)
    
    # split train and test data
    model_data = model_data.values.astype(np.float32)
    # Scale predictors.
    scaler = MinMaxScaler(feature_range=(0, 1))
    model_data[:, 0:3] = scaler.fit_transform(model_data[:, 0:3])
    
    # Pull out 90% for training. Ensure data is shuffled. 
    full_train, full_test = train_test_split(model_data, train_size=.9)
    
    predictor_train = full_train[:, 0:3]
    predictor_test = full_test[:, 0:3]
    target_train = full_train[:, 3]    
    target_test = full_test[:, 3]
    
    # Encode the targets
    encoder = OneHotEncoder()
    target_train = encoder.fit_transform(
        target_train.astype(np.int8).reshape((-1,1))
    )
    target_test = encoder.fit_transform(
        target_test.astype(np.int8).reshape((-1,1))
    )
    
    #sampler = SamplerFactory.get_instance(sample_method, ratio=1)
    #res_predictor_train, res_target_train = sampler.fit_sample(
    #    predictor_train, target_train
    #)
    
    res_predictor_train = predictor_train
    res_target_train = target_train
    
    inputs = Input(shape=(3,))
    x = Dense(64, activation='relu')(inputs)
    predictions = Dense(2, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=predictions)
    
    model.compile(
        loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy']
    )
    history = model.fit(
        res_predictor_train, res_target_train, 
        epochs=100, verbose=1, batch_size=128, 
        validation_data=(predictor_test, target_test)
    )
