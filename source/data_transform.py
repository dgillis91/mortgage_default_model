# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 21:20:35 2019

@author: dgill
"""

import os
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
    write_performance_counts(output, loan_dict)

def parse_performance_record_to_tuple(record):
    return (record[0], record[10], record[15])

def update_loan(delinquency_status, foreclosure_date, loan):
    if loan.is_trackable_payment_status(delinquency_status):
        loan.increment_payment_status_count(delinquency_status)
    if foreclosure_date != '':
        loan.is_foreclosed = True

def write_performance_counts(path, data):
    with open(path, 'w') as perf:
        perf.writelines([
            '{}\n'.format(loan.as_data_record()) for loan_number, loan in data.items()
        ])
    
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
    extract_performance_counts(performance_path, output_path)