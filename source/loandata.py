# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 20:15:45 2019

@author: dgill
"""

from collections import defaultdict

class Loan:
    __payment_status = set(list(range(4)))
    def __init__(self, loan_identifier):
        self._loan_identifier = loan_identifier
        self._payment_status_count = defaultdict(int)
        self._is_foreclosed = False
        
    @property
    def loan_identifier(self):
        return self._loan_identifier
    @property
    def is_foreclosed(self):
        return self._is_foreclosed
    @is_foreclosed.setter
    def is_foreclosed(self, value):
        self._is_foreclosed = value
    
    def is_trackable_payment_status(self, status):
        return status in Loan.__payment_status
    
    def increment_payment_status_count(self, status):
        self._payment_status_count[str(status)] += 1
        
    def as_data_record(self, sep='|'):
        return sep.join([
            self._loan_identifier, str(self._is_foreclosed)
        ] + [v for k, v in self._payment_status_count.items()])
    
    def __ne__(self, other):
        return not self.__eq__(other)
    def __eq__(self, other):
        return self.__cmp__(other)
    def __cmp__(self, other):
        return self._loan_identifier == other._loan_identifier
    def __hash__(self, other):
        return hash(self._loan_identifier)