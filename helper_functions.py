# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def get_x_y(df):
    '''
    Function to split dataframe into features and target
    
    Input: Cleaned and Prepped DataFrame
    
    Output: X features and Y target DataFrames
    '''
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]
    y.columns = ['target']
    return X, y



def scaling(X_train, X_test):
    '''
    Function to apply standard scaling post train_test_split to numerical features
    
    Input: X_train and X_test datasets
    
    Output: Standard Scaled X_train and X_test datasets
    '''
    
    ss = StandardScaler()
    # Fit Transform and replace on X_train
    numeric_train = X_train.iloc[:, 26:33]
    X_train.drop(columns = ['Neuroticism_score', 'Extraversion_score', 'Openness_score', 'Agreeableness_score',
                            'Conscientiousness_score', 'Impulsiveness', 'Sensation_seeing'], inplace = True)
    scaled_train = pd.DataFrame(ss.fit_transform(numeric_train))
    scaled_train.columns = numeric_train.columns
    scaled_train.index = numeric_train.index
    X_train = X_train.merge(scaled_train, right_index = True, left_index = True)
    
    # Transform and replace on X_test
    numeric_test = X_test.iloc[:, 26:33]
    X_test.drop(columns = ['Neuroticism_score', 'Extraversion_score', 'Openness_score', 'Agreeableness_score',
                            'Conscientiousness_score', 'Impulsiveness', 'Sensation_seeing'], inplace = True)
    scaled_test = pd.DataFrame(ss.transform(numeric_test))
    scaled_test.columns = numeric_test.columns
    scaled_test.index = numeric_test.index
    X_test = X_test.merge(scaled_test, right_index = True, left_index = True)
    
    return X_train, X_test



def smote_train(X_train, y_train):
    '''
    Function to apply SMOTE
    
    Input: Training features and target
    
    Output: Balanced training features and target
    '''
    
    smote = SMOTE()
    X_train_resampled, y_train_resampled = smote.fit_sample(X_train, y_train)
    
    return X_train_resampled, y_train_resampled