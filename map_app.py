import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import numpy as np
import joblib



# Load the trained model
model = joblib.load('xgb_best_model.joblib')

# Load the preprocessed dataset
data = pd.read_csv('combined.csv')

# Define the user interface
st.set_page_config(page_title="Heart Disease Prediction", page_icon=":heart:")
from PIL import Image

with st.sidebar:
    selected = option_menu('MENU',
                          
                          ['About',
                           'Heart Disease Prediction',
                           'Contact'],
                          icons=['activity','heart','person'],
                          default_index=0)


if (selected == 'Heart Disease Prediction'):
    # Set title
    st.title('Heart Disease Prediction')

    # Convert the 'Age' column to numeric
    data['Age'] = pd.to_numeric(data['Age'], errors='coerce')

    # Define the input fields in Streamlit
    age = st.slider('Age', min_value=int(data['Age'].min()), max_value=int(data['Age'].max()), value=int(data['Age'].median()))
    sex = st.selectbox('Sex', data['Sex'].unique())

    # Replace the chest_pain_type_mapping with full text representation
    chest_pain_type_dict = {
    'ATA': 'Atypical Angina',
    'NAP': 'Non-Anginal Pain',
    'ASY': 'Asymptomatic',
    'TA': 'Typical Angina'
    }
    chest_pain_type = st.selectbox('Chest Pain Type', data['ChestPainType'].unique(), format_func=lambda x: chest_pain_type_dict[x])
    
    # Convert 'RestingBP' column to a NumPy array
    resting_bp_values = data['RestingBP'].values

    resting_bp = st.slider('Resting Blood Pressure', min_value=int(resting_bp_values.min()), max_value=int(resting_bp_values.max()), value=int(np.median(resting_bp_values)))
    # Convert 'Cholesterol' column to a NumPy array
    cholesterol_values = data['Cholesterol'].values

    cholesterol = st.slider('Cholesterol', min_value=int(cholesterol_values.min()), max_value=int(cholesterol_values.max()), value=int(np.median(cholesterol_values)))
    fasting_bs = st.selectbox('Fasting Blood Sugar', data['FastingBS'].unique())
    resting_ecg = st.selectbox('Resting ECG', data['RestingECG'].unique())
    # Convert 'MaxHR' column to a NumPy array
    max_hr_values = data['MaxHR'].values

    max_hr = st.slider('Maximum Heart Rate', min_value=int(max_hr_values.min()), max_value=int(max_hr_values.max()), value=int(np.median(max_hr_values)))
    exercise_angina = st.selectbox('Exercise-Induced Angina', data['ExerciseAngina'].unique())
    # Convert 'Oldpeak' column to a NumPy array
    oldpeak_values = data['Oldpeak'].values

    oldpeak = st.slider('ST Depression Induced by Exercise Relative to Rest', min_value=float(oldpeak_values.min()), max_value=float(oldpeak_values.max()), value=float(np.median(oldpeak_values)))
    st_slope = st.selectbox('ST Slope', data['ST_Slope'].unique())

    # Map the input values to preprocessed dataset values
    sex_mapping = {'M': 0, 'F': 1}
    chest_pain_type_mapping = {val: idx for idx, val in enumerate(data['ChestPainType'].unique())}
    fasting_bs_mapping = {val: idx for idx, val in enumerate(data['FastingBS'].unique())}
    resting_ecg_mapping = {val: idx for idx, val in enumerate(data['RestingECG'].unique())}
    exercise_angina_mapping = {val: idx for idx, val in enumerate(data['ExerciseAngina'].unique())}
    st_slope_mapping = {val: idx for idx, val in enumerate(data['ST_Slope'].unique())}

    sex = sex_mapping[sex]
    chest_pain_type = chest_pain_type_mapping[chest_pain_type]
    fasting_bs = fasting_bs_mapping[fasting_bs]
    resting_ecg = resting_ecg_mapping[resting_ecg]
    exercise_angina = exercise_angina_mapping[exercise_angina]
    st_slope = st_slope_mapping[st_slope]

    # Create a feature vector from the input values
    input_data = np.array([[age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs,
                            resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]])

    # Define the prediction button
    if st.button('Predict Heart Disease'):
        # Make the prediction
        prediction = model.predict(input_data)
        
        # Display the prediction result
        if prediction[0] == 1:
            st.write('There is a heart disease.')
        else:
            st.write('There is no heart disease.')

def app():
    st.title("Contact")

    st.write('''
    ----
    *Thank you for using the Heart Disease Prediction using ML app! If you have any feedback or suggestions, please feel free to reach out to us.*

    **Contact Information:**

    **Email:** olaoguntamilore@gmail.com

    **Github** https://github.com/tamilore3000

    
    ''')

if (selected == 'About'):

    st.title("About")
    st.write('''
    ----
    

    *Welcome to the Heart Disease Prediction using ML app! This project aims to provide a simple yet effective way to predict the presence of heart disease in patients.*

    *The dataset used for this project was sourced from the cleveland heart disease dataset UCI repository. We made use of common and traditional machine learning techniques and steps, including data preprocessing, feature selection, and model training. Three algorithms were used in this project, namely Artificial Neural Networks (ANN), Logistic Regression, and Random Forest.*

    *To select the best model, we performed hyperparameter tuning on all three models, and ultimately selected the ANN model for deployment.*

    *We believe that this app can be a valuable tool in the healthcare industry, providing doctors and medical professionals with an efficient way to predict the presence of heart disease in their patients.*

    *Thank you for using the Heart Disease Prediction using ML app. If you have any feedback or suggestions, please feel free to reach out to us.*


    ''')

    # Button to go to the "Contact" page
    if st.button("Contact Us"):
        app()

if (selected == 'Contact'):
    app()