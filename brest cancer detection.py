# -*- coding: utf-8 -*-
"""
Created on Sat May  1 23:11:39 2021

@author: karan
"""

#Importing the libraries and dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Importing dataset
dataset = pd.read_csv('data.csv')


#Data Exploration
dataset.head()
dataset.shape
dataset.info()

dataset.select_dtypes(include='object').columns
len(dataset.select_dtypes(include='object').columns)

dataset.select_dtypes(include=['float64','int64']).columns
len(dataset.select_dtypes(include=['float64','int64']).columns)


# statistical summary
dataset.describe()

dataset.columns


"""Dealing with the missing values"""
dataset.isnull().values.any()

dataset.isnull().values.sum()

dataset.columns[dataset.isnull().any()]
len(dataset.columns[dataset.isnull().any()])

dataset['Unnamed: 32'].count()
dataset = dataset.drop(columns='Unnamed: 32')

dataset.shape
dataset.isnull().values.any()

"""Dealing with categorical data"""

dataset.select_dtypes(include='object').columns

dataset['diagnosis'].unique()
dataset['diagnosis'].nunique()

# one hot encoding
dataset = pd.get_dummies(data=dataset, drop_first=True)

dataset.head()


"""Data Visualization"""
sns.countplot(dataset['diagnosis_M'], label='Count')
plt.show()

# B (0) values
(dataset.diagnosis_M == 0).sum()

# M (1) values
(dataset.diagnosis_M == 1).sum()


"""Correlation matrix and heatmap"""
dataset_2 = dataset.drop(columns='diagnosis_M')
dataset_2.head()

dataset_2.corrwith(dataset['diagnosis_M']).plot.bar(
    figsize=(20,10), title = 'Correlated with diagnosis_M', rot=45, grid=True
)

# Correlation matrix
corr = dataset.corr()
corr

# heatmap
plt.figure(figsize=(100,50))
sns.heatmap(corr, annot=True)


"""Splitting the dataset train and test set"""
dataset.head()

# matrix of features / independent variables
x = dataset.iloc[:, 1:-1].values

# target variable / dependent variable
y = dataset.iloc[:, -1].values


from sklearn.model_selection import  train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

x_train.shape
x_test.shape
y_train.shape
y_test.shape

"""Feature scaling"""

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

"""Building the model"""

##Logistic regression

from sklearn.linear_model import LogisticRegression
classifir_lr = LogisticRegression(random_state=0)

classifir_lr.fit(x_train, y_train)

y_pred = classifir_lr.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)


results = pd.DataFrame([['Logistic Regression', acc, f1, prec, rec]],
               columns = ['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall'])

results

cm = confusion_matrix(y_test, y_pred)
print(cm)

"""Cross validation"""
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=classifir_lr, X=x_train, y=y_train, cv=10)

print("Accuracy is {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f} %".format(accuracies.std()*100))

"""Random forest"""
from sklearn.ensemble import RandomForestClassifier

classifier_rm = RandomForestClassifier(random_state=0)
classifier_rm.fit(x_train, y_train)

y_pred = classifier_rm.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

model_results = pd.DataFrame([['Random forest', acc, f1, prec, rec]],
               columns = ['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall'])

results = results.append(model_results, ignore_index=True)

results


cm = confusion_matrix(y_test, y_pred)
print(cm)

"""Cross validation"""
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=classifier_rm, X=x_train, y=y_train, cv=10)

print("Accuracy is {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f} %".format(accuracies.std()*100))


"""Randomized Search to find the best parameters (Logistic regression)"""
from sklearn.model_selection import RandomizedSearchCV

parameters = {'penalty':['l1', 'l2'],
              'C':[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
              'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
              }



random_search = RandomizedSearchCV(estimator=classifir_lr, param_distributions=parameters, n_iter=5, 
                                   scoring='roc_auc', n_jobs = -1, cv=5, verbose=3)


random_search.fit(x_train, y_train)

random_search.best_estimator_
random_search.best_score_
random_search.best_params_


"""Final model (Logistic regression)"""
from sklearn.linear_model import LogisticRegression
classifir = LogisticRegression(C=1.5, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l1',
                   random_state=0, solver='saga', tol=0.0001, verbose=0,
                   warm_start=False)
classifir.fit(x_train, y_train)

y_pred = classifir.predict(x_test)


acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

model_results = pd.DataFrame([['Final Logistic Regression', acc, f1, prec, rec]],
               columns = ['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall'])

results = results.append(model_results, ignore_index = True)
results

"""cross validation"""

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=classifir, X=x_train, y=y_train, cv=10)

print("Accuracy is {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f} %".format(accuracies.std()*100))