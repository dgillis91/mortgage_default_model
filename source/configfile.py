# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 22:29:26 2019

@author: dgill
"""

import os
import json
from dirutil import project_directory


def get_config(filename='default'):
    config_file = filename + '.json'
    config_path = os.path.join(project_directory(), 'config', config_file)
    with open(config_path, 'r') as cfg_file:
        config = json.loads(cfg_file.read())
    return config
