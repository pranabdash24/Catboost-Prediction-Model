import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
import pickle

# Load your trained model (ensure this path is correct)
model = pickle.load(open('model.pkl', 'rb'))

st.title("Gym Membership Activity Prediction")

# Input fields for the features
pool_visits = st.number_input("Enter the number of pool visits", min_value=0, step=1)
mem_length = st.number_input("Enter the length of membership (in days)", min_value=0, step=1)
mem_category = st.selectbox("Select membership category", ['peak', 'off-peak', 'others'])
age = st.number_input("Enter the user's age", min_value=0, step=1)
gender = st.selectbox("Select gender", ['Male', 'Female'])
class_bookings = st.number_input("Enter the number of class bookings", min_value=0, step=1)
alt_bookings = st.number_input("Enter the number of alternative bookings", min_value=0, step=1)
oth_bookings = st.number_input("Enter the number of other bookings", min_value=0, step=1)
squash_bookings = st.number_input("Enter the number of squash bookings", min_value=0, step=1)
price_level = st.selectbox("Select price level", ['student', 'alumni', 'others', 'community', 'staff', 'junior'])

# Prediction button
if st.button("Predict"):
    # Create a DataFrame from the inputs
    input_data = pd.DataFrame({
        'pool_visits': [pool_visits],
        'mem_length': [mem_length],
        'mem_category': [mem_category],
        'age': [age],
        'gender': [gender],
        'class_bookings': [class_bookings],
        'alt_bookings': [alt_bookings],
        'oth_bookings': [oth_bookings],
        'squash_bookings': [squash_bookings],
        'price_level': [price_level]
    })

    # Perform the prediction
    prediction = model.predict(input_data)[0]

    # Output the prediction
    if prediction == 1:
        st.success("The prediction is: Active")
    else:
        st.warning("The prediction is: Inactive")
