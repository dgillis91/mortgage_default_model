# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 22:21:47 2019

@author: dgill
"""

import numpy as np

import os
from configfile import get_config
from dirutil import project_directory

from keras.models import Model
from keras.layers import Dense, BatchNormalization, Input

from sklearn.model_selection import train_test_split

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

class SamplerFactory:
    def get_instance(sample_method, *args, **kwargs):
        if sample_method == 'over':
            return SMOTE(*args, **kwargs)
        if sample_method == 'under':
            return RandomUnderSampler(*args, **kwargs)
        else:
            raise ValueError('invalid parameter: {}'.format(sample_method))

if __name__ == '__main__':
    project_path = project_directory()
    config = get_config()
    train_pct = .9
    
    data = np.loadtxt(
        os.path.join(project_path, config['diw_path'], 'model_data.csv'),
        delimiter=config['data_sep']
    )
    
    n_predictors = len(data[0]) - 2
    predictors = data[:, :n_predictors]
    targets = data[:, n_predictors:]
    
    pred_train, pred_test, target_train, target_test = train_test_split(
        predictors, targets, train_size=0.9        
    )

    sampler = SamplerFactory.get_instance('under', ratio=1, return_indices=True)
    _, __, res_indices = sampler.fit_sample(
        pred_train, target_train
    )

    res_train_pred, res_train_target = pred_train[res_indices], target_train[res_indices]

    inputs = Input(shape=(predictors.shape[1],))
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(2, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=predictions)
    
    model.compile(
        optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy']
    )
    model.fit(
        res_train_pred, res_train_target,
        epochs=20, verbose=1, batch_size=256,
        validation_data=(pred_test, target_test)
    )
