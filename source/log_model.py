# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 19:41:06 2019

@author: dgill
"""

import os

from keras.models import Model
from keras.layers import Dense, BatchNormalization, Input
from keras.optimizers import SGD

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder

from sklearn.metrics import classification_report, auc, roc_curve

import numpy as np
import pandas as pd

from dirutil import project_directory
from configfile import get_config
from transformer import clean_nulls, SamplerFactory


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
    
    sampler = SamplerFactory.get_instance(sample_method, ratio=.8)
    res_predictor_train, res_target_train = sampler.fit_sample(
        predictor_train, target_train
    )
    
    inputs = Input(shape=(4,))
    x = Dense(1, activation='sigmoid')(inputs)
    model = Model(inputs=inputs, outputs=x)

    model.compile(
        loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy']
    )
    history = model.fit(
        res_predictor_train, res_target_train, 
        epochs=32, verbose=1, batch_size=64,
        validation_data=(predictor_test, target_test)
    )

    p = model.predict(predictor_test)
    p_nominal = [1 if x > .5 else 0 for x in p]

    print(classification_report(target_test, p_nominal))
    print('FEATURE IMPORTANCES')
    weights = model.get_weights()[0] * res_predictor_train.std()
    c_predictors = config['predictors']
    for header, importance in zip(c_predictors, weights):
        print('[+] {}: {:.4f}'.format(header, importance[0]))

    import matplotlib.pyplot as plt
    plt.plot(history.history['loss'])
    plt.show()
    plt.plot(history.history['acc'])
    plt.show()
    fpr, tpr, _ = roc_curve(target_test, p_nominal)
    print('AUC')
    print(auc(fpr, tpr))
