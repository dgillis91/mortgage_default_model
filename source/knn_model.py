# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 19:41:06 2019

@author: dgill
"""

import os

from sklearn.model_selection import train_test_split

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report

import numpy as np
import pandas as pd

from dirutil import project_directory
from configfile import get_config

from collections import defaultdict


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

    # Clean nulls and map f/c stat to bits
    clean_nulls(model_data, predictors)
    mapping = {True: 1, False: 0}
    model_data[target] = model_data[target].map(mapping)
    
    # split train and test data
    model_data = model_data.values.astype(np.float32)

    knn_models = defaultdict(dict)
    outfile_path = os.path.join(project_path, 'analysis', 'knn_analysis.csv')
    with open(outfile_path, 'w') as grid_search_out:
        grid_search_out.write('ratio,neighbor_count,precision_non_default,' +
                              'recall_non_default,precision_default,recall_default\n')
        sampling_ratios = np.linspace(.1, 1, 10)
        for ratio in sampling_ratios:
            full_train, full_test = train_test_split(model_data, shuffle=True, test_size=.1)

            predictor_train = full_train[:, 0:4]
            predictor_test = full_test[:, 0:4]
            target_train = full_train[:, 4]
            target_test = full_test[:, 4]

            sampler = SamplerFactory.get_instance(sample_method, ratio=ratio if ratio < 1 else 1)
            res_predictor_train, res_target_train = sampler.fit_sample(
                predictor_train, target_train
            )

            for neighbor_count in range(3, 20):
                classifier = KNeighborsClassifier(n_neighbors=neighbor_count)
                classifier.fit(res_predictor_train, res_target_train)

                p = classifier.predict(predictor_test)

                report = classification_report(target_test, p, output_dict=True)
                results = '{},{},{},{},{},{}\n'.format(
                    ratio, neighbor_count,
                    report['0.0']['precision'], report['0.0']['recall'],
                    report['1.0']['precision'], report['1.0']['recall']
                )
                print(ratio)
                grid_search_out.write(results)
                knn_models[ratio][neighbor_count] = classifier