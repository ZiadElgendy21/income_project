import pandas as pd 
import streamlit as st 

st.markdown("<h1> <center> Income Prediction DataSet </center> </h1>",unsafe_allow_html= True)
df = pd.read_csv("E:/Mid_Project_Data_Science/income/Sources/adult.csv")
st.markdown('<a href= "https://www.kaggle.com/datasets/wenruliu/adult-income-dataset/data"> <center> Link to DataSet </center> </a>',unsafe_allow_html= True)
st.write(df)