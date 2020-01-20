# -*- coding: utf-8 -*-
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.around(np.mean(np.abs((y_true - y_pred) / y_true)) * 100,4)

def mean_absolute_QuartoDecimo(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_pred=y_pred[:,3:10]
    y_true=y_true[:,3:10]


    return np.around(np.mean(np.abs((y_true - y_pred) / y_true)) * 100,4)