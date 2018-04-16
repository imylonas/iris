"""
Created on Mon Apr 16 10:48:17 2018

@author: yannis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier


"""Import data!! """
dataset=pd.read_csv('Iris.csv')

"""I seperate independent from dependent values """
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,5].values

""" Here I encode categorial variables """
labelencoder_y=LabelEncoder()
y_le=labelencoder_y.fit_transform(y)

"""Ι create test and train set """
x_train, x_test, y_train, y_test=train_test_split(x,y_le,test_size=0.25,random_state=0)
regressor=LinearRegression()
regressor.fit(x_train,y_train)

""" Scatter Plot"""
grr=pd.scatter_matrix(dataset, alpha=0.2, figsize=(15, 15), marker='o',
                   hist_kwds={'bins':20}, s=60)

""" k-nearest method to train our model"""
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)

"""test set """
y_pred=knn.predict(x_test)

print("Test set score:{:.2f}".format(np.mean(y_pred==y_test)))

print("Test set score: {:.2f}".format(knn.score(x_test,y_test)))
 
