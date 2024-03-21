import os
import json
import numpy as np
import pandas as pd
from math import factorial

from dataio.utils import load_XY
import warnings
warnings.filterwarnings("ignore")

def combination(n,r):
    '''Compute # possible combinations, i.e. choose r from n w/o repetition and order does NOT matter.'''
    out = factorial(n) / (factorial(r) * factorial(n-r))
    return out

def permutation(n,r, allow_repetition=True):
    '''Compute # possible combinations, i.e. choose r from n w/o or w repetition and order DOES matter.'''
    if allow_repetition:
        out = n**r
    else:
        out = factorial(n) / factorial(n-r)
    return out





def construct_global_space(ds_name, global_space_pth, y_type):
    '''Construct a pseudo global space defined by the input x.
        
    params:
        X: <pandas.DataFrame>
    '''

    if os.path.exists(global_space_pth):
        print('Loading global space from : %s'% global_space_pth)
        return global_space_pth
    print('Global space file not exists : %s'% global_space_pth)
    # construct global x
    X, _ = load_XY(ds_name=ds_name, as_matrix=False, y_type=y_type)
    fake_x_rng_dict = {}
    for feature in X.columns:
        _min = X[feature].min()
        _max = X[feature].max()
        _vals = np.unique(X[feature])
        _intervals =  [ _vals[i]-_vals[i-1] for i in range(1, len(_vals))]
        _intervals.sort()
        _intvl = _intervals[0]
        fake_x_rng_dict[feature] = {'type':'list_range', 'val':{'min':_min, 'max':_max, 'interval':_intvl}}

    with open(global_space_pth, 'w') as json_file:
        json.dump(fake_x_rng_dict, json_file)
    print('Saved global space range of dataset %s to %s'%(ds_name, global_space_pth))


    return  global_space_pth