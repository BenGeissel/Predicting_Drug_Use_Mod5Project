# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import itertools
import scikitplot as skplt
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
import grid_search_functions



def run_logreg(X_train, X_test, y_train, y_test, C, penalty):
    '''
    Function to fit Logistic Regression Model and predict on test set
    
    Input: Train and Test Data. Optimized C and Penalty parameters
    
    Output: Dataframe with metric scores. Confusion Matrix and ROC Curve plots.
    '''
    
    # Print test data balances
    y_pos = y_test.target.value_counts()[1]
    drug_user_percent = round(y_pos/ len(y_test), 2)
    print(f'Drug user percent: {drug_user_percent * 100}%')
    print()
    print('Logisitic Regression Results:')
    
    # Fit model and get predictions
    model = LogisticRegression(C = C, penalty = penalty, fit_intercept = False, solver = 'liblinear')
    X_train_res, y_train_res = smote_train(X_train, y_train)
    model_fit = model.fit(X_train_res, y_train_res)
    y_hat_test = model.predict(X_test)
    
    # Calculate metrics
    prec = precision_score(y_test, y_hat_test)
    recall = recall_score(y_test, y_hat_test)
    acc = accuracy_score(y_test, y_hat_test)
    f1 = f1_score(y_test, y_hat_test)
    print()
    print("Model Metrics:")
    print(f"Precision: {prec}")
    print(f"Recall: {recall}")
    print(f"Accuracy: {acc}")
    print(f"F1_Score: {f1}")
    
    # Plot Confusion Matrix
    skplt.metrics.plot_confusion_matrix(y_test, y_hat_test, figsize = (4,4))
    plt.ylim([1.5, -.5])
    plt.tight_layout()
    plt.show()
    
    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    AUC = auc(fpr, tpr)
    scores = [prec, recall, acc, f1, AUC]
    print()
    print(f'AUC: {AUC}')
    plt.plot(fpr, tpr, lw = 2, label = 'ROC Curve', color = 'orange')
    plt.plot([0,1], [0,1], lw = 2, linestyle = '--', color = 'r')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC Curve and AUC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.tight_layout()
    plt.show()
    
    print('-'*75)
    
    return pd.DataFrame({
        'Model' : ['Logistic_Regression' for i in scores],
        'Metric' : ['Precision', 'Recall', 'Accuracy', 'F1_Score', 'AUC'],
        'Score' : scores})



def run_NB_Gaussian(X_train, X_test, y_train, y_test):
    '''
    Function to fit Naive Bayes Gaussian Model and predict on test set
    
    Input: Train and Test Data
    
    Output: Dataframe with metric scores. Confusion Matrix and ROC Curve plots.
    '''
    
    print('Naive Bayes - Gaussian Results:')
        
    # Fit model and get predictions
    model = GaussianNB()
    X_train_res, y_train_res = smote_train(X_train, y_train)
    model_fit = model.fit(X_train_res, y_train_res)
    y_hat_test = model.predict(X_test)
    
    # Calculate metrics
    prec = precision_score(y_test, y_hat_test)
    recall = recall_score(y_test, y_hat_test)
    acc = accuracy_score(y_test, y_hat_test)
    f1 = f1_score(y_test, y_hat_test)
    print()
    print("Model Metrics:")
    print(f"Precision: {prec}")
    print(f"Recall: {recall}")
    print(f"Accuracy: {acc}")
    print(f"F1_Score: {f1}")
    
    # Plot Confusion Matrix
    skplt.metrics.plot_confusion_matrix(y_test, y_hat_test, figsize = (4,4))
    plt.ylim([1.5, -.5])
    plt.tight_layout()
    plt.show()
    
    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    AUC = auc(fpr, tpr)
    scores = [prec, recall, acc, f1, AUC]
    print()
    print(f'AUC: {AUC}')
    plt.plot(fpr, tpr, lw = 2, label = 'ROC Curve', color = 'orange')
    plt.plot([0,1], [0,1], lw = 2, linestyle = '--', color = 'r')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC Curve and AUC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.tight_layout()
    plt.show()
    
    print('-'*75)
    
    return pd.DataFrame({
        'Model' : ['NB_Gaussian' for i in scores],
        'Metric' : ['Precision', 'Recall', 'Accuracy', 'F1_Score', 'AUC'],
        'Score' : scores})



def run_knn(X_train, X_test, y_train, y_test, k):
    '''
    Function to fit K-Nearest Neighbors Model and predict on test set
    
    Input: Train and Test Data. Optimized k parameter.
    
    Output: Dataframe with metric scores. Confusion Matrix and ROC Curve plots.
    '''
    
    print('K-Nearest Neighbors Results:')
    
    # Fit model and get predictions
    model = KNeighborsClassifier(n_neighbors = k)
    X_train_res, y_train_res = smote_train(X_train, y_train)
    model_fit = model.fit(X_train_res, y_train_res)
    y_hat_test = model.predict(X_test)

    # Calculate metrics
    prec = precision_score(y_test, y_hat_test)
    recall = recall_score(y_test, y_hat_test)
    acc = accuracy_score(y_test, y_hat_test)
    f1 = f1_score(y_test, y_hat_test)
    print()
    print("Model Metrics:")
    print(f"Precision: {prec}")
    print(f"Recall: {recall}")
    print(f"Accuracy: {acc}")
    print(f"F1_Score: {f1}")
    
    # Plot Confusion Matrix
    skplt.metrics.plot_confusion_matrix(y_test, y_hat_test, figsize = (4,4))
    plt.ylim([1.5, -.5])
    plt.tight_layout()
    plt.show()
    
    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    AUC = auc(fpr, tpr)
    scores = [prec, recall, acc, f1, AUC]
    print()
    print(f'AUC: {AUC}')
    plt.plot(fpr, tpr, lw = 2, label = 'ROC Curve', color = 'orange')
    plt.plot([0,1], [0,1], lw = 2, linestyle = '--', color = 'r')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC Curve and AUC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.tight_layout()
    plt.show()
    
    print('-'*75)
    
    return pd.DataFrame({
        'Model' : ['KNN' for i in scores],
        'Metric' : ['Precision', 'Recall', 'Accuracy', 'F1_Score', 'AUC'],
        'Score' : scores})



def run_decision_tree(X_train, X_test, y_train, y_test):
    '''
    Function to fit Decision Tree Model and predict on test set
    
    Input: Train and Test Data.
    
    Output: Dataframe with metric scores. Confusion Matrix and ROC Curve plots.
    '''
    
    print('Decision Tree Results:')
    
    # Fit model and get predictions
    model = DecisionTreeClassifier()
    X_train_res, y_train_res = smote_train(X_train, y_train)
    model_fit = model.fit(X_train_res, y_train_res)
    y_hat_test = model.predict(X_test)
    
    # Calculate metrics
    prec = precision_score(y_test, y_hat_test)
    recall = recall_score(y_test, y_hat_test)
    acc = accuracy_score(y_test, y_hat_test)
    f1 = f1_score(y_test, y_hat_test)
    print()
    print("Model Metrics:")
    print(f"Precision: {prec}")
    print(f"Recall: {recall}")
    print(f"Accuracy: {acc}")
    print(f"F1_Score: {f1}")
    
    # Plot Confusion Matrix
    skplt.metrics.plot_confusion_matrix(y_test, y_hat_test, figsize = (4,4))
    plt.ylim([1.5, -.5])
    plt.tight_layout()
    plt.show()
    
    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    AUC = auc(fpr, tpr)
    scores = [prec, recall, acc, f1, AUC]
    print()
    print(f'AUC: {AUC}')
    plt.plot(fpr, tpr, lw = 2, label = 'ROC Curve', color = 'orange')
    plt.plot([0,1], [0,1], lw = 2, linestyle = '--', color = 'r')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC Curve and AUC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.tight_layout()
    plt.show()
    
    print('-'*75)
    
    return pd.DataFrame({
        'Model' : ['Decision_Tree' for i in scores],
        'Metric' : ['Precision', 'Recall', 'Accuracy', 'F1_Score', 'AUC'],
        'Score' : scores})



def run_random_forest(X_train, X_test, y_train, y_test, n):
    '''
    Function to fit Random Forest Model and predict on test set
    
    Input: Train and Test Data. Optimized n parameter.
    
    Output: Dataframe with metric scores. Confusion Matrix and ROC Curve plots.
    '''
    
    print('Random Forest Results:')
    
    # Fit model and get predictions
    model = RandomForestClassifier(n_estimators = n)
    X_train_res, y_train_res = smote_train(X_train, y_train)
    model_fit = model.fit(X_train_res, y_train_res)
    y_hat_test = model.predict(X_test)
    
    # Calculate metrics
    prec = precision_score(y_test, y_hat_test)
    recall = recall_score(y_test, y_hat_test)
    acc = accuracy_score(y_test, y_hat_test)
    f1 = f1_score(y_test, y_hat_test)
    print()
    print("Model Metrics:")
    print(f"Precision: {prec}")
    print(f"Recall: {recall}")
    print(f"Accuracy: {acc}")
    print(f"F1_Score: {f1}")
    
    # Plot Confusion Matrix
    skplt.metrics.plot_confusion_matrix(y_test, y_hat_test, figsize = (4,4))
    plt.ylim([1.5, -.5])
    plt.tight_layout()
    plt.show()
    
    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    AUC = auc(fpr, tpr)
    scores = [prec, recall, acc, f1, AUC]
    print()
    print(f'AUC: {AUC}')
    plt.plot(fpr, tpr, lw = 2, label = 'ROC Curve', color = 'orange')
    plt.plot([0,1], [0,1], lw = 2, linestyle = '--', color = 'r')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC Curve and AUC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.tight_layout()
    plt.show()
    
    print('-'*75)
    
    return pd.DataFrame({
        'Model' : ['Random_Forest' for i in scores],
        'Metric' : ['Precision', 'Recall', 'Accuracy', 'F1_Score', 'AUC'],
        'Score' : scores})



def run_svc(X_train, X_test, y_train, y_test):
    '''
    Function to fit Support Vector Machine Model and predict on test set
    
    Input: Train and Test Data.
    
    Output: Dataframe with metric scores. Confusion Matrix and ROC Curve plots.
    '''
    
    print('SVC Results:')
    
    # Fit model and get predictions
    model = SVC(probability = True)
    X_train_res, y_train_res = smote_train(X_train, y_train)
    model_fit = model.fit(X_train_res, y_train_res)
    y_hat_test = model.predict(X_test)

    # Calculate metrics
    prec = precision_score(y_test, y_hat_test)
    recall = recall_score(y_test, y_hat_test)
    acc = accuracy_score(y_test, y_hat_test)
    f1 = f1_score(y_test, y_hat_test)
    print()
    print("Model Metrics:")
    print(f"Precision: {prec}")
    print(f"Recall: {recall}")
    print(f"Accuracy: {acc}")
    print(f"F1_Score: {f1}")
    
    # Plot Confusion Matrix
    skplt.metrics.plot_confusion_matrix(y_test, y_hat_test, figsize = (4,4))
    plt.ylim([1.5, -.5])
    plt.tight_layout()
    plt.show()
    
    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    AUC = auc(fpr, tpr)
    scores = [prec, recall, acc, f1, AUC]
    print()
    print(f'AUC: {AUC}')
    plt.plot(fpr, tpr, lw = 2, label = 'ROC Curve', color = 'orange')
    plt.plot([0,1], [0,1], lw = 2, linestyle = '--', color = 'r')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC Curve and AUC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.tight_layout()
    plt.show()
    
    print('-'*75)
    
    return pd.DataFrame({
        'Model' : ['SVC' for i in scores],
        'Metric' : ['Precision', 'Recall', 'Accuracy', 'F1_Score', 'AUC'],
        'Score' : scores})



def run_linear_svc(X_train, X_test, y_train, y_test):
    '''
    Function to fit Linear Support Vector Machine Model and predict on test set
    
    Input: Train and Test Data.
    
    Output: Dataframe with metric scores. Confusion Matrix.
    '''
    
    print('Linear SVC Results:')
    
    # Fit model and get predictions
    model = LinearSVC()
    X_train_res, y_train_res = smote_train(X_train, y_train)
    model_fit = model.fit(X_train_res, y_train_res)
    y_hat_test = model.predict(X_test)
    
    # Calculate metrics
    prec = precision_score(y_test, y_hat_test)
    recall = recall_score(y_test, y_hat_test)
    acc = accuracy_score(y_test, y_hat_test)
    f1 = f1_score(y_test, y_hat_test)
    print()
    print("Model Metrics:")
    print(f"Precision: {prec}")
    print(f"Recall: {recall}")
    print(f"Accuracy: {acc}")
    print(f"F1_Score: {f1}")
    
    # Plot Confusion Matrix
    skplt.metrics.plot_confusion_matrix(y_test, y_hat_test, figsize = (4,4))
    plt.ylim([1.5, -.5])
    plt.tight_layout()
    plt.show()
    
    scores = [prec, recall, acc, f1]

    print('-'*75)
    
    return pd.DataFrame({
        'Model' : ['Linear_SVC' for i in scores],
        'Metric' : ['Precision', 'Recall', 'Accuracy', 'F1_Score'],
        'Score' : scores})



def run_sgd(X_train, X_test, y_train, y_test):
    '''
    Function to fit Stochastic Gradient Descent Model and predict on test set
    
    Input: Train and Test Data.
    
    Output: Dataframe with metric scores. Confusion Matrix and ROC Curve plots.
    '''
    
    print('SGD Results:')
    
    # Fit model and get predictions
    model = SGDClassifier()
    X_train_res, y_train_res = smote_train(X_train, y_train)
    clf = model.fit(X_train_res, y_train_res)
    calibrator = CalibratedClassifierCV(clf, cv='prefit')
    model_fit = calibrator.fit(X_train_res, y_train_res)
    y_hat_test = model.predict(X_test)
    
    # Calculate metrics
    prec = precision_score(y_test, y_hat_test)
    recall = recall_score(y_test, y_hat_test)
    acc = accuracy_score(y_test, y_hat_test)
    f1 = f1_score(y_test, y_hat_test)
    print()
    print("Model Metrics:")
    print(f"Precision: {prec}")
    print(f"Recall: {recall}")
    print(f"Accuracy: {acc}")
    print(f"F1_Score: {f1}")
    
    # Plot Confusion Matrix
    skplt.metrics.plot_confusion_matrix(y_test, y_hat_test, figsize = (4,4))
    plt.ylim([1.5, -.5])
    plt.tight_layout()
    plt.show()
    
    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, calibrator.predict_proba(X_test)[:,1])
    AUC = auc(fpr, tpr)
    scores = [prec, recall, acc, f1, AUC]
    print()
    print(f'AUC: {AUC}')
    plt.plot(fpr, tpr, lw = 2, label = 'ROC Curve', color = 'orange')
    plt.plot([0,1], [0,1], lw = 2, linestyle = '--', color = 'r')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC Curve and AUC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.tight_layout()
    plt.show()
    
    print('-'*75)
    
    return pd.DataFrame({
        'Model' : ['SGD' for i in scores],
        'Metric' : ['Precision', 'Recall', 'Accuracy', 'F1_Score', 'AUC'],
        'Score' : scores})