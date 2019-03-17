# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 11:30:50 2019

@author: dgill
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from configfile import get_config
from dirutil import project_directory

def get_foreclosure_counts(performance_df):
    return performance_df[
        ['is_foreclosed','loan_identifier']
    ].groupby('is_foreclosed').count()

if __name__ == '__main__':
    config = get_config()
    project_path = project_directory()
    
    performance_path = os.path.join(
        project_path, config['landing_path'], 'Performance.txt'
    )
    analysis_path = os.path.join(project_path, 'analysis')
    
    acquisition_data = pd.read_csv(
        os.path.join(project_path, config['diw_path'], 'diw.txt'),
        sep=config['data_sep']        
    )

    # Foreclosure frequency
    cnt = sns.countplot(x='is_foreclosed', data=acquisition_data)
    cnt.get_figure().savefig(os.path.join(analysis_path, 'fc_stat_freq'))
    plt.close('all')
    
    # Credit Score by foreclosure status
    credit_score_box = sns.boxenplot(
        x='is_foreclosed', y='borrower_credit_score_at_origination',
        data=acquisition_data
    )
    credit_score_box.get_figure().savefig(os.path.join(
        analysis_path, 'fc_cs'
    ))
    plt.close('all')
    
    sc = sns.scatterplot(x='borrower_credit_score_at_origination',
         y='original_combined_ltv', hue='is_foreclosed', 
         data=acquisition_data[[
             'borrower_credit_score_at_origination', 'original_combined_ltv',
             'is_foreclosed'
         ]]
    )
    sc.get_figure().savefig(
        os.path.join(analysis_path, 'credit_vs_combined_ltv.png')
    )
    plt.close('all')
    
    pair_cols = [
        'original_interest_rate', 'original_upb', 'original_loan_term',
        'original_ltv', 'original_combined_ltv', 'original_dti',
        'borrower_credit_score_at_origination'      
    ]
    pair_data = acquisition_data[pair_cols + ['is_foreclosed']].dropna().sample(10000)
    pair = sns.pairplot(pair_data, hue='is_foreclosed', vars=pair_cols)
    pair.savefig(
        os.path.join(analysis_path, 'pairs')        
    )
    
    
    
    
    
    
    
    