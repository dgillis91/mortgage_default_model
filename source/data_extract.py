# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 19:55:13 2019

@author: dgill
"""

import os
from zipfile import ZipFile
from dirutil import project_directory
import json

class Extractor:
    def __init__(self):
        pass
    
class ZipExtractor:
    __zip_ext = '.zip'
    
    def unzip_all(self, from_path, to_path=None, remove_zips=False):
        for filename in os.listdir(from_path):
            if filename.endswith(ZipExtractor.__zip_ext):
                filepath = os.path.join(from_path, filename)
                self._unzip(filepath, to_path)
                if remove_zips:
                    os.remove(filename)
    
    def _unzip(self, filename, to_path):
        with ZipFile(filename, 'r') as zf:
            zf.extractall(to_path)
        
    def _remove(self, filename):
        os.remove(filename)
        
if __name__ == '__main__':
    with open(os.path.join(project_directory(), 'config', 'default.json'), 'r') as cfg_file:
        config = json.loads(cfg_file.read())
    e = ZipExtractor()
    p_path = project_directory()
    zips = os.path.join(p_path, config['zip_path'])
    print(zips)
    landing = os.path.join(p_path, config['landing_path'])
    e.unzip_all(zips, landing, config['remove_zips'])