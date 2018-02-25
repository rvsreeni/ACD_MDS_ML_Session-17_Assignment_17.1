#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 09:23:55 2018

@author: macuser
"""

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
from pandas import Series, DataFrame
from pylab import rcParams

from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus


url= "https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv"
titanic = pd.read_csv(url)

titanic.columns = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
titanic['Sex_label'], _ = pd.factorize(titanic['Sex'])
titanic['Age'].fillna(0, inplace=True)

# select features
Y = titanic['Survived']
X = titanic[['Pclass','Sex_label','Age','SibSp','Parch','Fare']]
#print(X.isnull().sum())

# split data randomly into 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(
X, Y, test_size = 0.3, random_state = 100)       

# train the decision tree (criteria entropy)
dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=6, 
random_state=50)
print('\n Decision tree (criteria entropy)')
print(dtree.fit(X_train, y_train))

# use the model to make predictions with the test data
y_pred = dtree.predict(X_test)

# how did our model perform?
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))

# train the decision tree (criteria gini)
dtree = tree.DecisionTreeClassifier(criterion='gini', max_depth=6, 
random_state=50)
print('\n Decision tree (criteria gini)')
print(dtree.fit(X_train, y_train))

# use the model to make predictions with the test data
y_pred = dtree.predict(X_test)

# how did our model perform?
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))