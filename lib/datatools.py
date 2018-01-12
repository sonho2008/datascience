
#python 3
### This contains tools for data analysis with light import: 
###     numpy, pandas, matplotlib
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import int, bytes, chr

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def variance_test(df, xcol, ycol):
    """ For equal variance test. Return a pair of standard deviations of y 
    variable for lower half and upper half range of x values.
    
    df : a pandas dataframe;    
    xcol, ycol: strings representing column names of df;
    """
    xmed = df[xcol].median()
    df1 = df.loc[df[xcol] <= xmed]
    df2 = df.loc[df[xcol] >= xmed]
    return df1[ycol].std(), df2[ycol].std()

def mean_fill(df, default = 0):
    """ fill missing values for df with the mean of each column. df types must 
    be all numerical.
    """
    for c in df.columns:
        m = df[c].mean()
        if np.isnan(m): m = 0
        df.loc[:,c] = df.loc[:,c].fillna(value=m)
    return df

def median_fill(df, default = 0):
    """ fill missing values for df with the median of each column. df types 
    must be all numerical. 
    """
    for c in df.columns:
        m = df[c].median()
        if np.isnan(m): m = 0
        df.loc[:,c] = df.loc[:,c].fillna(value=m)
    return df

def standardize(df, epsilon = 0.001):
    """ Standardize each column of df and return a pair (newDf, W)
    where newDf is the standardized dataframe, and W is a dataframe
    containing means and standard deviations of the old df.
    """
    means = df.mean()
    stds = df.std() + epsilon
    newDf = (df - means)/stds
    W = pd.concat([means,stds], axis=1)
    W.columns = ['mean','std']
    return (newDf, W)

def numerical(df):
    """ Return a sub-dataframe of df containing only columns of numeric dtype. 
    """
    return df.select_dtypes(include = [np.number])

def non_numerical(df):    
    """ Return a sub-dataframe of df containing only columns of non-numeric 
    dtype. 
    """
    return df.select_dtypes(exclude = [np.number])

def squared_distance(x, y):
    """ Return squared Euclidean distance between points x and y. """
    diff = np.subtract(x,y).flatten()
    return np.sum(np.square(diff))

def get_top2_values(L):
    """ L must be a numpy array with some np.float dtype. """
    v1 = max(L[0],L[1])
    v2 = min(L[0],L[1])
    for i in range(2,len(L)):
        v = L[i]
        if v>v1:
            v2 = v1
            v1 = v
        elif v>v2:
            v2 = v
    return [v1,v2]

def clean_obj_column(ds, 
                     badList=['\r']):
    """ ds is a pandas Series """
    for c in badList:
        ds = ds.str.replace(c, '')
    return ds

def clean_df_objs(df):
    """ """
    for c in df.columns:
        if df[c].dtype == np.object:
            df[c] = clean_obj_column(df[c])
    return df
            
def read_exported_csv(filepath,
                      idcol = 'Id',
                      na_vals=['-',' - ','  -  ']):
    """ Read exported csv file, set id column as index whenever available. 
    """        
    df = pd.read_csv(filepath,
                     na_values=na_vals,
                     low_memory=False)
    df = clean_df_objs(df)
    if idcol in df.columns:
        df[idcol] = df[idcol].astype(np.int)
        df = df.set_index(idcol, verify_integrity=True)
    return df

def write_csv(df, filepath, 
              enc='utf-8'):
    """Standardized way to write csv for PPT using pandas. Make sure to
    never write the default RangeIndex.
    """
    if pd.isnull(df.index.name): #index has no name
        df.to_csv(filepath, index=False, encoding=enc)
    else:
        df.to_csv(filepath, index=True, encoding=enc)
    return

        
#### test code
#print('OK')

