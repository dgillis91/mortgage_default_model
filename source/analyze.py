# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 11:30:50 2019

@author: dgill
"""

import pandas as pd
import matplotlib.pyplot as plt

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
    
    performance_data = pd.read_csv(performance_path, config['data_sep'])
    foreclosure_counts = get_foreclosure_counts(performance_data)
    
    fc_count_axes = plt.subplot(2, 1, 1)
    foreclosure_counts.plot(
        kind='bar', title='foreclosure_counts', ax=fc_count_axes
    )