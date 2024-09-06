import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
 
 

# Load preprocessed features, model, scaler, and columns
with open("E:/Mid_Project_Data_Science/income/Pages/preprocessed_features.pkl", 'rb') as f:
    Features = pickle.load(f)

with open("E:/Mid_Project_Data_Science/income/Pages/logistic_regression_model.pkl", 'rb') as f:
    model = pickle.load(f)

with open("E:/Mid_Project_Data_Science/income/Pages/scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

with open("E:/Mid_Project_Data_Science/income/Pages/columns.pkl", 'rb') as f:
    columns = pickle.load(f)

# Function to preprocess input data
def preprocess_input(data):
    input_df = pd.DataFrame([data])
    input_df = pd.get_dummies(input_df)
    print("Columns in input_df after one-hot encoding:", input_df.columns)
    # Align the input DataFrame with the training data columns
    input_df = input_df.reindex(columns=columns, fill_value=0)
    print("Columns in input_df after reindexing:", input_df.columns)
    # Ensure no unexpected columns are included
    assert 'income' not in input_df.columns, "Unexpected 'income' column found in input data"
    input_scaled = scaler.transform(input_df)
    return input_scaled

# Streamlit UI
st.title('Income Prediction App')

# Input fields
workclass = st.selectbox('Workclass', ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
education_num = st.slider('Education Number', 1, 16, 10)
marital_status = st.selectbox('Marital Status', ['married', 'single', 'divorced'])
occupation = st.selectbox('Occupation', ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
gender = st.selectbox('Gender', ['Male', 'Female'])
native_country = st.selectbox('Native Country', ['United-States', 'Cambodia', 'England', 'Honduras', 'Hungary', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Ireland', 'Italy', 'Mexico', 'Malaysia', 'Philippines', 'Poland', 'Portugal', 'Puerto-Rico', 'Jamaica', 'Vietnam', 'Columbia', 'Egypt', 'Thailand', 'Trinadad&Tobago', 'Peru', 'Outlying-US(Guam-USVI-etc)', 'Taiwan', 'France', 'Dominican-Republic', 'El-Salvador', 'Canada', 'Germany', 'Iran', 'Mexico', 'New-Zealand', 'Scotland'])

# Prepare input data
input_data = {
    'workclass': workclass,
    'educational-num': education_num,
    'marital-status': marital_status,
    'occupation': occupation,
    'gender': gender,
    'native-country': native_country
}

# Predict button
if st.button('Predict'):
    input_scaled = preprocess_input(input_data)
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.write('The predicted income is: >50K')
    else:
        st.write('The predicted income is: <=50K')
