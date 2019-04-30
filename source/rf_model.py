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
from keras.utils import plot_model #TODO: Try this out

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder

from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import pandas as pd

from dirutil import project_directory
from configfile import get_config

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

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
    def get_instance(sample_method, *args, **kwargs):
        if sample_method == 'over':
            return SMOTE(*args, **kwargs)
        if sample_method == 'under':
            return RandomUnderSampler(*args, **kwargs)
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
    config = get_config('random_forest')
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
    #del performance_data
    
    # Apply Transforms
    
    ##md
    
    # Clean nulls and map f/c stat to bits
    clean_nulls(model_data, predictors)
    mapping = {True: 1, False: 0}
    model_data[target] = model_data[target].map(mapping)
    
    # split train and test data
    model_data = model_data.values.astype(np.float32)
    # Scale predictors.
    scaler = MinMaxScaler(feature_range=(0, 1))
    model_data[:, 0:3] = scaler.fit_transform(model_data[:,0:3])
    
    # Pull out 90% for training. Ensure data is shuffled. 
    full_train, full_test = train_test_split(model_data, train_size=.9)
   



    predictor_train = full_train[:, 0:3]
    predictor_test = full_test[:, 0:3]
    target_train = full_train[:, 3]    
    target_test = full_test[:, 3]


    #rf
    org =lm
    Data1=trainModel(full_train)
    TestData1=trainModel(fulltest)


    #rf
    y_train1=Data1.int_rt
    Data1.drop('int_rt',axis=1,inplace=True)
    x_train1=Data1

    regr_rf=RandomForestRegressor(max_depth=8)
    regr_rf.fit(x_train1,y_train1)

    computations(regr_rf,x_train1,y_train1)

    y_test1=TestData1.int_rt
    TestData1.drop('int_rt',axis=1,inplace=True)
    x_test1=TestData1
    computations(regr_rf,x_test1,y_test1)

    plt.scatter(regr_rf.predict(x_train1),regr_rf.predict(x_train1)-y_train1,c='b',s=40,alpha=0.5)
    plt.scatter(regr_rf.predict(x_test1),regr_rf.predict(x_test1)-y_test1,c="g",s=40)
    plt.hlines(y=0,xmin=2,xmax=10)
    plt.title('Residual plot using training(blue) and test(green) data')
    plt.ylabel('Residuals')
    
    #computations(regr_rf,aredictor_train,target_train)





    # Encode the targets
    encoder = OneHotEncoder()
    target_train = encoder.fit_transform(
        target_train.astype(np.int8).reshape((-1,1))
    )
    target_test = encoder.fit_transform(
        target_test.astype(np.int8).reshape((-1,1))
    )
    
    #sampler = SamplerFactory.get_instance(sample_method, ratio=1)
    #res_predictor_train, res_target_train = sampler.fit_sampla(
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

    
