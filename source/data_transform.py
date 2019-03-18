# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 21:20:35 2019

@author: dgill
"""

import os
import pandas as pd

from loandata import Loan
from configfile import get_config
from dirutil import project_directory

def extract_performance_counts(performance_path, output, sep='|'):
    loan_dict = {}
    with open(performance_path, 'r') as performance:
        for record in performance:
            parsed_record = record.split(sep)
            loan_number, delinquency_status, foreclosure_date = \
                parse_performance_record_to_tuple(parsed_record)
            if loan_number in loan_dict:
                loan = loan_dict[loan_number]
            else:
                loan = Loan(loan_number)
                loan_dict[loan_number] = loan
            update_loan(delinquency_status, foreclosure_date, loan)
    write_performance_counts(
        output, loan_dict, sep, [
            'loan_identifier', 'is_foreclosed',
            'no_0_past_due_months', 'no_1_past_due_months',
            'no_2_past_due_months', 'no_3_past_due_months'
        ]
    )

def parse_performance_record_to_tuple(record):
    return (record[0], record[10], record[15])

def update_loan(delinquency_status, foreclosure_date, loan):
    if loan.is_trackable_payment_status(delinquency_status):
        loan.increment_payment_status_count(delinquency_status)
    if foreclosure_date != '':
        loan.is_foreclosed = True

def write_performance_counts(path, data, sep, headers):
    with open(path, 'w') as perf:
        perf.write(sep.join(headers) + '\n')
        perf.writelines([
            '{}\n'.format(loan.as_data_record()) 
            for loan_number, loan in data.items()
        ])
        
def merge_acquisition_and_performance(acquisition_path, performance_path, 
                                      acq_headers, perf_headers, sep='|'):
    acquisition_data = pd.read_csv(
        acquisition_path, sep=sep, names=acq_headers
    )
    perf_data = pd.read_csv(
        performance_path, sep=sep, names=perf_headers        
    )
    merged_data = pd.merge(
        acquisition_data, performance_data, on='loan_identifier'        
    )
    return merged_data
    
if __name__ == '__main__':
    config = get_config()
    project_path = project_directory()
    
    performance_file = 'Performance_2007Q4.txt'
    performance_path = os.path.join(
        project_path, config['landing_path'], performance_file
    )
    output_path = os.path.join(
        project_path, config['landing_path'], 'Performance.txt'
    )
    acquisition_path = os.path.join(
        project_path, config['landing_path'], 'Acquisition_2007Q4.txt'        
    )
    
    extract_performance_counts(performance_path, output_path)
    
    merged_acq_and_perf = merge_acquisition_and_performance(
        acquisition_path, output_path, 
        config['acquisition_headers'], config['parsed_performance_headers']        
    )
    merged_acq_and_perf.to_csv(
        os.path.join(project_path, config['diw_path'], 'diw.txt'),
        sep=config['data_sep'], index=False
    )
    
    