# Import necessary libraries
import pandas as pd

def prep_data(df):
    '''
    Function to dummy variable all categorical columns and reorder columns into useful order
    
    Input: Cleaned dataframe
    
    Output: Dataframe with dummy variables in correct order
    '''
    
    # Dummy variables for categorical data columns
    df = pd.get_dummies(df, columns = ['Age', 'Gender', 'Education_level', 'Country', 'Ethnicity'],
                                    drop_first = True)
    
    # Reorder dataframe columns
    df = df[['Age_25-34', 'Age_35-44', 'Age_45-54', 'Age_55-64', 'Age_65+', 'Gender_Male', 'Education_level_17',
        'Education_level_18', 'Education_level_< 16', 'Education_level_Associates degree',
        'Education_level_Bachelors degree', 'Education_level_Doctorate degree', 'Education_level_Masters degree',
        'Education_level_Some college', 'Country_Canada', 'Country_Ireland', 'Country_New Zealand', 'Country_Other',
        'Country_UK', 'Country_USA', 'Ethnicity_Black', 'Ethnicity_Mixed-Black/Asian', 'Ethnicity_Mixed-White/Asian',
        'Ethnicity_Mixed-White/Black', 'Ethnicity_Other', 'Ethnicity_White', 'Neuroticism_score', 'Extraversion_score',
        'Openness_score', 'Agreeableness_score', 'Conscientiousness_score', 'Impulsiveness', 'Sensation_seeing',
        'Semer_fake_drug', 'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caffeine', 'Cannabis', 'Chocolate', 'Cocaine',
        'Crack', 'Ecstacy', 'Heroin', 'Ketamine', 'Legal_highs', 'LSD', 'Meth', 'Mushrooms', 'Nicotine']]
    
    return df