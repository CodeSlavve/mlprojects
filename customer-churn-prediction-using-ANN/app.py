import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import pickle
import os
from tensorflow.keras.models import load_model

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.h5') # Or 'models/model.h5' if it's in a subfolder

# Load the model safely
model = load_model(MODEL_PATH)  

# Load encoders and scaler
with open('ohe_geo.pkl', 'rb') as file:
    ohe_geo = pickle.load(file)

with open('ohe_gender.pkl', 'rb') as file:
    ohe_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# Streamlit app

st.title('Customer Churn Prediction')

# User Input
geography = st.selectbox('Geography', ohe_geo. categories_[0])
gender = st.selectbox('Gender', ohe_gender.categories_[0])
age = st.slider('Age', 18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])

# prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
})

# OHE 'Geography' and 'Gender'
geo_encoded = ohe_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))

gender_encoded = ohe_gender.transform([[gender]])
gender_encoded_df = pd.DataFrame(gender_encoded, columns=ohe_gender.get_feature_names_out(['Gender']))

# Combine OHE cols with input data
input_data = pd.concat(
    [
        input_data.reset_index(drop=True),
        geo_encoded_df.reset_index(drop=True),
        gender_encoded_df.reset_index(drop=True)
    ],
    axis=1
)

# Scale input
input_data_scaled = scaler.transform(input_data)

# Prediction Churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likey to churn.')
