'''
Created on 21 Oct 2013

@author: rstones
'''
import inspect
import numpy as np

class DataModel(object):
    '''
    Generic super-class containing functions that allow parameters to be saved to file and instances of the class to
    be reconstructed from the saved data.
    '''

    def __init__(self, fn=None):
        if fn:
            self.initialize_instance_from_params(self.read_params(fn))
        
    def save_params(self, fn):
        attributes = inspect.getmembers(self, lambda a:not(inspect.isroutine(a)))
        params = [a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))]
        with open(fn, 'w') as f:
            for p in params:
                if not isinstance(p[1], np.ndarray):
                    f.write(p[0] + ' ' + str(p[1]) + '\n')
                    
    def read_params(self, fn):
        params = {}
        with open(fn, 'r') as f:
            for line in f:
                if '[' not in line: # line does not contain list
                    key, value = line.split()
                    params[key] = float(value) if '.' in value else int(value)
                else: # deal with list params
                    key, value = line.split(' [')
                    value = value.strip(']\n')
                    value = value.split(', ')
                    value = [float(v) for v in value]
                    params[key] = value
                    
        return params
    
    def initialize_instance_from_params(self, params):
        for k in params:
            self.__dict__[k] = params[k]