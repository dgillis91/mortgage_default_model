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
     
class ArbitraryFileSampler:
    def __init__(self, from_path, to_path, sep='|'):
        self._from_path = from_path
        self._to_path = to_path
        self._sep = sep

    def sample(self):
        sampled = set()
        with open(self._from_path, 'r') as _in, open(self._to_path, 'w') as out:
            for line in _in:
                parsed = line.split(self._sep)
                if parsed[0] not in sampled:
                    out.write(line)
                    sampled.add(parsed[0])
            

if __name__ == '__main__':
    config_path = os.path.join(project_directory(), 'config', 'default.json')
    with open(config_path, 'r') as cfg_file:
        config = json.loads(cfg_file.read())
    e = ZipExtractor()
    p_path = project_directory()
    #zips = os.path.join(p_path, config['zip_path'])
    #print(zips)
    landing = os.path.join(p_path, config['landing_path'])
    perf_path = os.path.join(landing, 'Performance_2007Q4.txt')
    perf_arb_path = os.path.join(landing, 'arb_perf.txt')
    #e.unzip_all(zips, landing, config['remove_zips'])
    #sampler = ArbitraryFileSampler(perf_path, perf_arb_path)
    #sampler.sample()
    
    
    
    
        