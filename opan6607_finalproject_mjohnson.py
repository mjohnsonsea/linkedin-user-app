import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import streamlit as st

# Read CSV
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
ss = ss[['sm_li', 'income', 'education', 'parent', 'married', 'female', 'age']]

# Remove NA's
ss = ss.dropna()

# Target Vector
y = ss['sm_li']

# Feature Set
X = ss.drop(columns=['sm_li'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=206)

# Create logistic regression model
model = LogisticRegression(class_weight='balanced', random_state=206)

# Fit the model with the training data
model.fit(X_train, y_train)

# Streamlit App Header
st.write("# LinkedIn User Prediction Model")
st.write("Created by: Mike Johnson")

# Dictionary of income options
income_options = {
    "Less than $10,000": 1,
    "10 to under $20,000": 2,
    "20 to under $30,000": 3,
    "30 to under $40,000": 4,
    "40 to under $50,000": 5,
    "50 to under $75,000": 6,
    "75 to under $100,000": 7,
    "100 to under $150,000": 8,
    "$150,000 or more": 9
}

# Dictionary of education options
education_options = {
    "Less than high school": 1,
    "High school incomplete": 2,
    "High school graduate": 3,
    "Some college, no degree": 4,
    "Two-year associate degree": 5,
    "Bachelor’s degree": 6,
    "Some postgraduate or professional schooling": 7,
    "Postgraduate or professional degree": 8
}

# Dictionary of parent options
parent_options = {
    "Yes": 1,
    "No": 0
}

# Dictionary of marital options
marital_options = {
    "Married": 1,
    "Living with a Partner": 0,
    "Divorced": 0,
    "Separated": 0,
    "Widowed": 0,
    "Never been married": 0
}

# Dictionary of gender options
gender_options = {
    "Male": 0,
    "Female": 1,
    "Other": 2
}

# Create dropdown fields and slider for the inputs
income = st.selectbox('Income', options=list(income_options.keys()))
education = st.selectbox('Education', options=list(education_options.keys()))
parent = st.selectbox('Parent', options=list(parent_options.keys()))
married = st.selectbox('Marital Status', options=list(marital_options.keys()))
female = st.selectbox('Gender', options=list(gender_options.keys()))
age = st.slider('Age', min_value=18, max_value=97, value=18)

# Convert inputs to numerical values using the dictionaries
input_data = [income_options[income], 
              education_options[education], 
              parent_options[parent], 
              marital_options[married], 
              gender_options[female], 
              age]

# Convert to DataFrame for prediction
input_df = pd.DataFrame([input_data], columns=['income', 'education', 'parent', 'married', 'female', 'age'])

# Predict button
if st.button('Predict'):
    prediction = model.predict(input_df)
    prediction_proba = np.round(model.predict_proba(input_df)[:, 1] * 100).astype(int)
    
    st.markdown(f"Predicted LinkedIn User: **{'Yes' if prediction[0] == 1 else 'No'}**", unsafe_allow_html=True)
    st.markdown(f"Probability of being a LinkedIn User: **{prediction_proba[0]}%**", unsafe_allow_html=True)
