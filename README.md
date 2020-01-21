# **Predicting Drug Use through Classification Models**
# Module 5 Project
Predicting Drug Use as a binary outcome based on demographic and personality traits of a person.

## Ben Geissel

### Goals:
Run numerous classification models for each of 17 different drugs in the dataset to determine if a person has used the drug or not.
- Fit Numerous Classification Models (Logistic Regression, Decision Tree, KNN, SVM, etc...)
- Predict whether or not a certain person has used the drug
- Determine which models work best for each combination of drug and perfomance metric


### Responsibilities:
- ETL on drug use data
- Data exploration and visualizations
- Create helper functions
- Create grid search functions to optimize parameters for certain classification models
- Create functions to run each model and output confusion matrices and ROC curves
- Run all models on all drugs
- Determine best performing model for each combination of drug and performance metric

### Summary of Included Files:
The following files are included in the Github repository:
- drug_consumption_Mod5Project_Technical_Notebook_BenGeissel.ipynb
    - Jupyter Notebook for technical audience
    - PEP 8 Standards
    - Utilizes .py files for ETL, helper functions, grid search functions, and model fitting functions
    - Data importation, data cleaning, visualizations and charts
- cleaning.py
   - ETL function to clean raw dataset
- data_prep.py
   - Fit dummy variables
   - Reorder columns
- helper_functions.py
   - X, y split (features and target)
   - Standard Scaling
   - SMOTE sampling
- grid_search_functions.py
   - Grid search for C and Penalty hyperparameters for Logistic Regression
   - Grid search for k (n_neighbors) hyperparameter for KNN
   - Grid search for n (n_estimators) hyperparameter for Random Forest
- modeling_functions.py
   - Functions to fit each of the following models and plot confusion matrix and ROC curve
    - Logistic Regression
    - Naive Bayes (Gaussian)
    - KNN
    - Decision Tree
    - Random Forest
    - Support Vector Machine
    - Linear Support Vector Machine
    - Stochastic Gradient Descent
- Drug_Use_Classification_Mod5Project_BenGeissel.pdf
   - PDF presentation for project
   - Uses Cannabis and Heroin as case studies
- drug_consumption.txt
   - Original Dataset
- Drug_classifier_scores_df.csv
   - Cleaned Metric Scores Dataset (for backup)
- Data Visualizations Folder
    - Visuals used for presentation
- .gitignore
