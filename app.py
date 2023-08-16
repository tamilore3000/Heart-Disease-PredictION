import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the dataset
data = pd.read_csv('combined.csv')

# Convert 'Age' column to numeric type
data['Age'] = pd.to_numeric(data['Age'], errors='coerce')

# Load the trained model
model = joblib.load('mlp_best_model.joblib')

# Define the input fields in Streamlit
# Convert 'Age' column to a NumPy array
age_values = data['Age'].values
# Define the input fields in Streamlit
age = st.slider('Age', min_value=int(age_values.min()), max_value=int(age_values.max()), value=int(np.median(age_values)))
sex = st.selectbox('Sex', ['Male', 'Female'], index=0)
chest_pain_type = st.selectbox('Chest Pain Type', data['ChestPainType'].unique())
# Convert 'RestingBP' column to a NumPy array
resting_bp_values = data['RestingBP'].values

# Define the input fields in Streamlit
resting_bp = st.slider('Resting Blood Pressure', min_value=int(resting_bp_values.min()), max_value=int(resting_bp_values.max()), value=int(np.median(resting_bp_values)))

# Convert 'Cholesterol' column to a NumPy array
cholesterol_values = data['Cholesterol'].values

# Define the input fields in Streamlit
cholesterol = st.slider('Cholesterol', min_value=int(cholesterol_values.min()), max_value=int(cholesterol_values.max()), value=int(np.median(cholesterol_values)))

fasting_bs = st.selectbox('Fasting Blood Sugar', ['0', '1'], index=0)
# Convert 'MaxHR' column to a NumPy array
max_hr_values = data['MaxHR'].values

# Define the input fields in Streamlit
max_hr = st.slider('Maximum Heart Rate', min_value=int(max_hr_values.min()), max_value=int(max_hr_values.max()), value=int(np.median(max_hr_values)))

exercise_angina = st.selectbox('Exercise-Induced Angina', ['No', 'Yes'], index=0)
# Convert 'Oldpeak' column to a NumPy array
oldpeak_values = data['Oldpeak'].values

# Define the input fields in Streamlit
oldpeak = st.slider('ST Depression Induced by Exercise Relative to Rest', min_value=float(oldpeak_values.min()), max_value=float(oldpeak_values.max()), value=float(np.median(oldpeak_values)))

st_slope = st.selectbox('ST Slope', data['ST_Slope'].unique())
resting_ecg_0 = st.selectbox('Resting ECG 0', ['No', 'Yes'], index=0)
resting_ecg_1 = st.selectbox('Resting ECG 1', ['No', 'Yes'], index=0)
resting_ecg_2 = st.selectbox('Resting ECG 2', ['No', 'Yes'], index=0)

# Convert categorical variables to numeric values
sex = 0 if sex == 'Male' else 1
chest_pain_type = data[data['ChestPainType'] == chest_pain_type].index[0]
st_slope = data[data['ST_Slope'] == st_slope].index[0]
fasting_bs = int(fasting_bs)
exercise_angina = 0 if exercise_angina == 'No' else 1
resting_ecg_0 = 0 if resting_ecg_0 == 'No' else 1
resting_ecg_1 = 0 if resting_ecg_1 == 'No' else 1
resting_ecg_2 = 0 if resting_ecg_2 == 'No' else 1

# Create a feature vector
feature_vector = np.array([age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, max_hr, exercise_angina, oldpeak, st_slope, resting_ecg_0, resting_ecg_1, resting_ecg_2])

# Convert the feature vector to numeric type
feature_vector = feature_vector.astype(float)

# # Make a prediction
# prediction = model.predict([feature_vector])[0]

# # Display the prediction
# st.header('Heart Disease Prediction')
# if prediction == 0:
#     st.write('The model predicts that the individual does not have heart disease.')
# else:
#     st.write('The model predicts that the individual has heart disease.')
# Create a button
prediction_button = st.button('Predict')

# Check if the button is clicked
if prediction_button:
    # Make a prediction
    prediction = model.predict([feature_vector])[0]

    # Display the prediction
    st.header('Heart Disease Prediction')
    if prediction == 0:
        st.write('The model predicts that the individual does not have heart disease.')
    else:
        st.write('The model predicts that the individual has heart disease.')
