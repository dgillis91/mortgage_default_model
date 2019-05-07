# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 19:41:06 2019

@author: dgill
"""

import os

from keras.models import Model
from keras.layers import Dense, BatchNormalization, Input

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.metrics import classification_report

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
        sampler = SamplerFactory.__sampler.get(sample_method)
        if sampler is not None:
            return sampler(*args, **kwargs)
        else:
            raise ValueError('invalid parameter: {}'.format(sample_method))


if __name__ == '__main__':
    sample_method = 'under'
    project_path = project_directory()
    config = get_config('standard_model')
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
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # model_data[:, 0:3] = scaler.fit_transform(model_data[:, 0:3])
    
    # Pull out 90% for training. Ensure data is shuffled.
    full_train, full_test = train_test_split(model_data, shuffle=True, test_size=.1)
    
    predictor_train = full_train[:, 0:4]
    predictor_test = full_test[:, 0:4]
    target_train = full_train[:, 4]
    target_test = full_test[:, 4]

    scaler = MinMaxScaler(feature_range=(0, 1))
    predictor_test = scaler.fit_transform(predictor_test)
    predictor_train = scaler.fit_transform(predictor_train)
    
    sampler = SamplerFactory.get_instance(sample_method, ratio=1)
    res_predictor_train, res_target_train = sampler.fit_sample(
        predictor_train, target_train
    )
    
    inputs = Input(shape=(4,))
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=predictions)
    
    model.compile(
        loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy']
    )
    history = model.fit(
        res_predictor_train, res_target_train, 
        epochs=32, verbose=1, batch_size=64,
        validation_data=(predictor_test, target_test)
    )

    p = model.predict(predictor_test)
    p_nominal = [1 if x > .5 else 0 for x in p]

    print(classification_report(target_test, p_nominal))