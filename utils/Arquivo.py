# -*- coding: utf-8 -*-
import pandas as pd


def ler():
    dataset = pd.read_csv('../data/projeto base completa.csv',delimiter =';')
    return dataset