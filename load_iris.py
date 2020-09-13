# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 20:06:27 2020

@author: ferna
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np


from sklearn.datasets import load_iris
dataset = load_iris()
data = pd.DataFrame(dataset.data, dataset.target)
X, y = load_iris(return_X_y=True, as_frame=True)
features_name = dataset.target_names
print('\nloading Dataset... \n')


# Plotting dataset
from yellowbrick.features import pca_decomposition
print('loading plotting dataset... \n')
pca_decomposition(X,y, scale=True, projection=2, classes=features_name)


# Preprocessing data
from sklearn.preprocessing import StandardScaler
X_scale = StandardScaler().fit_transform(X)
print('loading Standardization... \n')


# Data splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2)
print('loading data splitting... \n')


# Learning algorithm
from sklearn.svm import SVC
model = SVC().fit(X_train,y_train)
y_pred = model.predict(X_test)
print('Training Accuracy : ', model.score(X_train,y_train))
print('Testing Accuracy : ', model.score(X_test,y_test))


# Hyperparameter Optimization
from sklearn.model_selection import GridSearchCV
gsc = GridSearchCV(SVC(), {
    'kernel': ['linear','poly','rbf','sigmoid'], 
    'gamma': ['scale','auto'], 
    'C': [0.1,1,10,100]}, 
    cv=5, return_train_score=True)
gsc.fit(X_train,y_train)
tuning = pd.DataFrame(gsc.cv_results_)


model = SVC(C=10, gamma='auto', kernel='rbf').fit(X_train,y_train)
y_pred = model.predict(X_test)
print('Training Accuracy : ', model.score(X_train,y_train))
print('Testing Accuracy : ', model.score(X_test,y_test))


# CV model
from sklearn.model_selection import cross_val_score, ShuffleSplit
cvs = cross_val_score(SVC(), X, y, cv=5)
print('\nCV-model... \n')
cvs.mean()


# Evaluate Model Performance
from sklearn.metrics import (balanced_accuracy_score, matthews_corrcoef, 
                             roc_auc_score, classification_report)
print('Balanced accuracy score : ' ,balanced_accuracy_score(y_test, y_pred))
print('MCC : ', matthews_corrcoef(y_test, y_pred))
#print('roc/auc score : ', roc_auc_score(y_test, y_pred, multi_class='ovr'))
print('Classification report : \n', classification_report(y_test, y_pred, 
                                                          target_names=features_name))


from yellowbrick.classifier import (confusion_matrix, ROCAUC, 
                                    precision_recall_curve)
print('\nloading plots...\n')
confusion_matrix(model, X_train, y_train, X_test, y_test)

roc = ROCAUC(model)
roc.fit(X_train, y_train)
roc.score(X_test, y_test)
roc.show()

precision_recall_curve(model, X_train, y_train, X_test, y_test)


from yellowbrick.classifier import class_prediction_error
class_prediction_error(model, X_train, y_train, X_test, y_test)








