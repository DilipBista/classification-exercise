#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns

# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# turn off pink boxes for demo
import warnings
warnings.filterwarnings("ignore")

# import our own acquire module
import acquire


# In[3]:


# create function prep_iris
def prep_iris(iris_df):
    iris_dummy_df = pd.get_dummies(iris_df[['species']], dummy_na = False, drop_first = [True])
    iris_df = pd.concat([iris_df, iris_dummy_df], axis = 1)    
    return iris_df


# In[4]:


# split data 
def split_iris(iris_df):
    iris_train, iris_test = train_test_split(iris_df, test_size = 0.2, random_state = 123)
    iris_train, iris_validate = train_test_split(iris_train, test_size = .3, random_state = 123)
    return iris_train, iris_validate, iris_test


# In[5]:



# Create function prep_titanic

def prep_titanic(titanic_df):
    '''
    This function will accept raw titanic data and prepare the titanic dataframe
    '''
    titanic_df = titanic_df.drop_duplicates()
    cols_to_drop = ['deck', 'embarked', 'class', 'passenger_id']
    titanic_df = titanic_df.drop(columns=cols_to_drop)
    titanic_dummy_df = pd.get_dummies(titanic_df[['sex', 'embark_town']], dummy_na = False, drop_first = [True, True])
    titanic_df = pd.concat([titanic_df, titanic_dummy_df], axis = 1)
    return titanic_df


# In[6]:


def split_titanic(titanic_df):
    '''
    Takes in a dataframe and returns train, validate, and test
    '''
    titanic_train, titanic_test = train_test_split(titanic_df, test_size = 0.2, stratify = titanic_df.survived, random_state = 123)
    titanic_train, titanic_validate = train_test_split(titanic_train, test_size = .3, random_state = 123, stratify = titanic_train.survived)
    return titanic_train, titanic_validate, titanic_test


# In[ ]:




