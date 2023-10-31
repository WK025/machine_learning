import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import plotly.express as px

# Read data
df = pd.read_csv("datasets/downsampled_smoking_driking_dataset.csv")
# Load saved model from pickle file
with open("saved_stacked_models/StackedPickleDrinking.pkl", 'rb') as file:
    pickle_drink_model = pickle.load(file)
with open("saved_stacked_models/StackedPickleSmoking.pkl", 'rb') as file:
    pickle_smoke_model = pickle.load(file)

# Important features
important_features_drk = ["gamma_GTP","age"]
important_features_smk = ["age","hemoglobin"]

# Boxplot for each target variable
for i in important_features_drk:
    plot_drk = px.box(df, x=df[i], y=df["DRK_YN"])
    plot_drk.update_layout(
    title=f'Box Plot of {i} vs {df["DRK_YN"]}',
    xaxis_title=i,
    yaxis_title="Drinking Status"
    )

for i in important_features_smk:
    plot_drk = px.box(df, x=df[i], y=df["SMK_stat_type_cd"])
    plot_drk.update_layout(
    title=f'Box Plot of {i} vs {df["SMK_stat_type_cd"]}',
    xaxis_title=i,
    yaxis_title="Smoking Status"
    )

# Streamlit Setup
# set title
st.title('Smoking and Drinking Status Prediction App')
# set description
st.write('Boxplots for Important Features with Drinking and Smoking Status')
# add header 2 through markdown
# st.markdown("## Smoking Status:")

col1, col2 = st.columns(2)

with col1:
   st.header("Important Feature with Drinking Status")
#    st.image("https://static.streamlit.io/examples/cat.jpg")
   st.plotly_chart(plot_drk)

with col2:
   st.header("Important Feature with Smoking Status")
#    st.image("https://static.streamlit.io/examples/dog.jpg")
   st.plotly_chart(plot_drk)

# add header 2 through markdown
st.markdown("## Target Variable Prediction")

st.markdown('### Input Parameters')

feature_vec = st.text_input("Input a comma-seperated list of features (20): ")
st.markdown('Example: List of features: 0,35,81,0.5,0.6,1,1,93,53,85,69,117,30,11.3,1,0.8,20,7,10,23.4')

x = feature_vec.split(",")
x = [float(i) for i in x] 
x = np.array(x).reshape(1, -1)


st.markdown('### Drinking Status Prediction')
y_drink_predict = pickle_drink_model.predict(x)[0]
if y_drink_predict==0:
    drink_res='Drinker'
elif y_drink_predict==1:
    drink_res='Non-drinker'
st.write(drink_res)

st.markdown('### Smoking Status Prediction')
y_smoke_predict = pickle_smoke_model.predict(x)[0]
if y_smoke_predict==0:
    smoke_res='Never Smoking'
elif y_smoke_predict==1:
    smoke_res='Used to smoke but quit'
elif y_smoke_predict==2:
    smoke_res="Still smoking"
st.write(smoke_res)