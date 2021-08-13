# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 21:43:40 2020

@author: myusuf
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

print(x)
print(y)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(x[:,1:3])
x[:, 1:3] = imputer.transform(x[:,1:3])

print(x)

#----------------------------------------------
# Encoding the categorial data 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
x= np.array(ct.fit_transform(x))

print(x)

#----------------------------------------------
# Encoding the dependent variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y= le.fit_transform(y)

print('The data transformed as follows')

print(y)

#----------------------------------------------
# Splitting the dataset into training data test data

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

print('see the data after splitting it')
print(x_train)
print('---------------------------------')
print('********************************')
print(x_test)
print(y_train)
print(y_test)

#----------------------------------------------
# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:,3:] = sc.fit_transform(x_train[:,3:])
x_test[:,3:]=sc.transform(x_test[:,3:])
print('---------------------------------')
print('********************************')
print(x_train)
print('---------------------------------')
print('********************************')
print(x_test)







