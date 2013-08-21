# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:29:10 2013

@author: Tom
"""

import numpy as np

def gen_target_function(d = 2, wgt = None):
    if wgt is None:
        wgt = np.random.rand(d + 1) * 2 - 1.0
    elif len(wgt) != (d + 1):
        print('{} elements in wgt {}.  {} elements expected.'.format(len(wgt), wgt, d + 1))
        return None
    def func(x):
        x1 = np.ones(len(x) + 1)
        x1[1:] = x
        return sum(np.array(x1, dtype = float) * wgt)
    return func

def gen_data_set(f, d = 2, n = 20, scale = 10):
    xs = (np.random.rand(n, d) - 0.5) * scale * 2.0
    return (xs, np.array([np.sign(f(x)) for x in xs]))
    
def run_pla(xs, ys, wgt = None, n_max_iter = 30, verbosity = 1):
    d = np.size(xs, axis = 1)
    if wgt is None:
        wgt = np.zeros(d + 1)
    elif len(wgt) != (d + 1):
        print('{} elements in initial guess wgt {}.  {} elements expected.'.format(len(wgt), wgt, d + 1))
        return None
        
    for i_iter in range(n_max_iter):
        if verbosity > 0:
            print('Iteration {}'.format(i_iter))
        f = gen_target_function(d = d, wgt = wgt)
        hs = [np.sign(f(x)) for x in xs]
        matched = hs == ys
        if all(matched):
            # found function that perfectly classifies the data set.
            if verbosity > 0:
                print('Classification function found in {} steps.'.format(i_iter))
            break
        else:
            x_i = xs[~matched][0]
            y_i = ys[~matched][0]
            h_i = hs[~matched][0]
            assert(h_i != y_i)
            if verbosity > 0:
                print('Mis-classified {} as {} instead of {}'.format(x_i, h_i, y_i))
            x1_i = np.ones(len(x) + 1)
            x1_i[1:] = x_i
            n_wgt = wgt + y_i * x1_i
            if verbosity > 0:
                print('old wgt {}, new wgt {}'.format(wgt, n_wgt))
            wgt = n_wgt
            
    return gen_target_function(wgt)