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
    age = st.slider('Age', min_value=0, max_value=150, value=int(data['Age'].median()))
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

    st.title("About Heart Disease Prediction Using Machine Learning")
    st.write('''
    ----
    ## Welcome to the "Heart Disease Prediction Using Machine Learning" App!

    This app is designed to provide a powerful tool for predicting the presence of heart disease in patients based on various health parameters. By leveraging machine learning techniques, I aim to assist medical professionals in making accurate predictions about heart health.

    ### Project Overview:

    In the pursuit of creating an effective heart disease prediction model, I have utilized a diverse dataset containing a wide range of health metrics. By analyzing and understanding the relationship between these metrics and the presence of heart disease, I've developed a robust machine learning model.

    ### Dataset Description:

    The dataset used in this project contains information about various health parameters of patients. Here's a breakdown of the dataset columns:

    - **Age:** Age of the patient [years]
    - **Sex:** Sex of the patient [Male, Female]
    - **ChestPainType:** Chest pain type [Typical Angina, Atypical Angina, Non-Anginal Pain, Asymptomatic]
    - **RestingBP:** Resting blood pressure [mm Hg]
    - **Cholesterol:** Serum cholesterol [mm/dl]
    - **FastingBS:** Fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
    - **RestingECG:** Resting electrocardiogram results [Normal: Normal, ST-T wave abnormality, Left Ventricular Hypertrophy (LVH)]
    - **MaxHR:** Maximum heart rate achieved [Numeric value between 60 and 202]
    - **ExerciseAngina:** Exercise-induced angina [Yes, No]
    - **Oldpeak:** ST depression induced by exercise relative to rest [Numeric value measured in depression]
    - **ST_Slope:** The slope of the peak exercise ST segment [Upsloping, Flat, Downsloping]
    - **HeartDisease:** Output class [1: heart disease, 0: Normal]

    ### Machine Learning Models:

    The "Heart Disease Prediction Using Machine Learning" app employs the following machine learning models:

    - **Multilayer Perceptron (MLP) Classifier:** A neural network model capable of learning complex relationships in the data.
    - **Random Forest:** A powerful ensemble model that combines multiple decision trees for accurate predictions.
    - **XGBoost:** An optimized gradient boosting algorithm for high-performance predictions.
    - **Support Vector Machine (SVM):** A model that finds the best hyperplane to separate data points into different classes.

    After rigorous evaluation and tuning, the best-performing model was selected and deployed to make predictions within this app.

    ### Future Enhancements:

    While the current version of the app is already a valuable tool, potential future enhancements could include:

    - Integration of additional health metrics for even more accurate predictions.
    - Incorporation of interpretability techniques to provide insights into the model's decision-making process.
    - User-friendly visualizations to better understand the importance of different features in predictions.
    - Real-time updates and feedback based on the latest research and medical advancements.

    Thank you for using the "Heart Disease Prediction Using Machine Learning" app. If you have any feedback, questions, or suggestions, please don't hesitate to reach out to me using the contact information provided.
    *About the Creator:*

    *The "Heart Disease Prediction Using Machine Learning" app was created by Tamilore Olaogun in 2023.*
    ''')

    # Button to go to the "Contact" page
    if st.button("Contact Us"):
        app()

if (selected == 'Contact'):
    app()