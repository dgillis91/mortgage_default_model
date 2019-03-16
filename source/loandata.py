# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 20:15:45 2019

@author: dgill
"""

class Loan:
    def __init__(self, loan_identifier):
        self._loan_identifier = loan_identifier
        self._payment_status_count = self._default_payment_status_count()
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
    
    def increment_payment_status_count(self, status):
        self._payment_status_count[status] += 1
        
    def _default_payment_status_count(self):
        counts = {}
        for i in range(4):
            counts[str(i)] = 0
    
    def __ne__(self, other):
        return not self.__eq__(other)
    def __eq__(self, other):
        return self.__cmp__(other)
    def __cmp__(self, other):
        return self._loan_identifier == other._loan_identifier
    def __hash__(self, other):
        return hash(self._loan_identifier)