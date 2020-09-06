# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import pandas as pd
from yellowbrick.datasets import load_mushroom
import seaborn as sn
import numpy as np

dataset = load_mushroom(return_dataset=True)
df = dataset.to_dataframe()
df.head()
X = df.drop(columns=['target'])
y = df['target']
print('\nDataset Mushroom\n')


# Preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()
ohe = OneHotEncoder(handle_unknown='ignore')
y_scale = le.fit_transform(y)
X_scale = ohe.fit_transform(X)

from yellowbrick.target import class_balance

class_balance(y_scale, labels=['edible','poisonous'])


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scale, y_scale, 
                                                    test_size=0.2, 
                                                    random_state=0)
print('Preprocessing\n')
print('X_train : ', X_train.shape, '\nX_test : ', X_test.shape, '\ny_train : ', 
      y_train.shape, '\ny_test : ', y_test.shape)



# Learning Algorithms
print('\nLearning Algorithms\n')
from sklearn.svm import SVC
'''
model = SVC(kernel='linear').fit(X_train, y_train)
y_pred = model.predict(X_test)
print('\nTraining Accuracy : ', model.score(X_train, y_train))
print('Test Accuracy : ', model.score(X_test, y_test))
'''


# Hyperparameters Optimization
'''
from sklearn.model_selection import GridSearchCV
print('\nHyperparameters Optimization...\n')
gsc = GridSearchCV(SVC(), {
    'C': [0.1, 1, 10, 100, 1000],
    'kernel': ['linear','poly','rbf','sigmoid'],
    'gamma': ['scale','auto']
    }, cv=5, return_train_score=True)

gsc.fit(X_train,y_train)
result = pd.DataFrame(gsc.cv_results_)
'''

model = SVC(kernel='rbf', C=100, gamma='auto').fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Training Accuracy : ', model.score(X_train, y_train))
print('Test Accuracy : ', model.score(X_test, y_test))



# Cross Validation Model
from sklearn.model_selection import (cross_val_score, StratifiedShuffleSplit)
from yellowbrick.model_selection import CVScores

cv = StratifiedShuffleSplit(n_splits=5, random_state=0)
cvs = cross_val_score(SVC(kernel='rbf', C=100, gamma='auto'), X_scale, 
                      y_scale, cv=cv, scoring='f1_macro')
print('\nCross Validation\n')
print('Cross Validation Score : ', cvs.mean())

cv_vis = CVScores(SVC(kernel='rbf', C=100, gamma='auto'), cv=cv, scoring='f1_macro')
cv_vis.fit(X_scale, y_scale)
cv_vis.show()
print('\nVisualization...')



# Scoring Estimator
print('\nScoring Estimator\n')
## Classification Report
from sklearn.metrics import (classification_report, confusion_matrix)

name = ['edible', 'poisonous']
cr = classification_report(y_test, y_pred, target_names=name)
print('Classification Report : \n', cr)


## Confusion Matrix
from yellowbrick.classifier import confusion_matrix

#matrix = confusion_matrix(y_test, y_pred)
cm = confusion_matrix(model, X_train, y_train , X_test, y_test, 
                      classes=['edible','poisonous'])
print('\nVisualization Matrix Confusion...')
#print('\nMatrix Confusion : \n', matrix)


## ROC/AUC
from yellowbrick.classifier import ROCAUC

roc = ROCAUC(model,micro=False, macro=False, per_class=False , 
             classes=['edible','poisonous'])
roc.fit(X_train, y_train)
roc.score(X_test, y_test,)
roc.show()
print('\nVisualization ROC/AUC...')


## Precision-Recall Curves
from yellowbrick.classifier import precision_recall_curve

pr = precision_recall_curve(model, X_train, y_train, X_test, y_test)
print('\nVisualization Precision-Recall Curves...')


## Class Prediction Error
from yellowbrick.classifier import class_prediction_error

vis = class_prediction_error(model, X_train, y_train, X_test, y_test, 
                             classes=['edible','poisonous'])
vis.show()
print('\nVisualization Class Prediction Error...')

















