# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
import helper_functions



def grid_search_CP(X_train, y_train):
    '''
    Function to grid search for best C and penalty for Logistic Regression
    
    Input: Training features and target
    
    Output: Optimized C and Penalty parameters for Logistic Regression
    '''
    
    penalty = ['l1', 'l2']
    C = np.arange(.1, 50, .5)
    hyperparameters = dict(C = C, penalty = penalty)
    lr = LogisticRegression()
    clf = GridSearchCV(lr, hyperparameters, cv = 5, verbose = 0)
    X_train_res, y_train_res = helper_functions.smote_train(X_train, y_train)
    grid = clf.fit(X_train_res, y_train_res)
    c = grid.best_estimator_.get_params()['C']
    p = grid.best_estimator_.get_params()['penalty']
    
    return c, p



def grid_search_neighbors(X_train, y_train):
    '''
    Function to grid search for best k value (# of neighbors for knn)
    
    Input: Training features and target
    
    Output: Optimized k (n_neighbors) parameter for KNN
    '''
    
    K = list(range(1, 9, 2))
    hyperparameters = dict(n_neighbors = K)
    knn = KNeighborsClassifier()
    clf = GridSearchCV(knn, hyperparameters, cv = 5, verbose = 0)
    X_train_res, y_train_res = helper_functions.smote_train(X_train, y_train)
    grid = clf.fit(X_train_res, y_train_res)
    k = grid.best_estimator_.get_params()['n_neighbors']
    
    return k



def grid_search_estimators(X_train, y_train):
    '''
    Function to grid search for best n value (# of estimators for random forest)
    
    Input: Training features and target
    
    Output: Optimized n (n_estimators) parameter for Random Forest
    '''
    
    N = list(range(70, 125, 5))
    hyperparameters = dict(n_estimators = N)
    rf = RandomForestClassifier()
    clf = GridSearchCV(rf, hyperparameters, cv = 5, verbose = 0)
    X_train_res, y_train_res = helper_functions.smote_train(X_train, y_train)
    grid = clf.fit(X_train_res, y_train_res)
    n = grid.best_estimator_.get_params()['n_estimators']
    
    return n