# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 22:29:26 2019

@author: dgill
"""

import os
import json

def get_config():
    config_path = os.path.join(project_directory(), 'config', 'default.json')
    with open(config_path, 'r') as cfg_file:
        config = json.loads(cfg_file.read())
    return config