#### Final Project
#### Mike Johnson

# Load libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Read csv
s = pd.read_csv('social_media_usage.csv')

# Create clean_sm function
def clean_sm(x):
    return np.where(x == 1, 1, 0)

# Create new dataframe
ss = s

# Create target column that indicates whether the individual uses LinkedIn
ss['sm_li'] = clean_sm(ss['web1h'])

# Transformations
ss['income'] = np.where(ss['income'] > 9, np.nan, ss['income']) # Above 9 is considered missing.
ss['education'] = np.where(ss['educ2'] > 8, np.nan, ss['educ2']) # Above 8 is considered missing.
ss['parent'] = np.where(ss['par'] == 1, True, False) # Binary
ss['married'] = np.where(ss['marital'] == 1, True, False) # Binary
ss['female'] = np.where(ss['gender'] == 2, True, False) # Binary
ss['age'] = np.where(ss['age'] > 98, np.nan, ss['age']) # Above 98 is considered missing.

# Select columns to include
ss = ss[['sm_li',
         'income',
         'education',
         'parent',
         'married',
         'female',
         'age']]

# Remove NA's
ss = ss.dropna()

# Target Vector
y = ss['sm_li']

# Feature Set
X = ss.drop(columns = ['sm_li'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    stratify = y,
                                                    test_size = 0.2, 
                                                    random_state = 206)

# Crete logistic regression model
model = LogisticRegression( class_weight = 'balanced', random_state = 206)

# Fit the model with the training data
model.fit(X_train, y_train)

# Test