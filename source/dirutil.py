# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 20:41:55 2019

@author: dgill
"""

import os

def project_directory():
    parent, folder_name = os.path.split(__file__)
    project_path = 'mortgage_default_model'
    
    while folder_name != project_path:
        parent, folder_name = os.path.split(parent)
    return os.path.join(parent, folder_name)