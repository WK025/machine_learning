import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

from pathlib import Path
path = Path(__file__).parent


# Read data
# df = pd.read_csv("./datasets/downsampled_dataset_after_feature_selection.csv")
df = pd.read_csv(path/"datasets/downsampled_dataset_after_feature_selection.csv")


# Streamlit Setup
# set title
st.title('Smoking and Drinking Status Prediction App')
# set description
st.markdown('## Boxplots for Important Features with Drinking and Smoking Status')


# Important features
important_features_drk = ["gamma_GTP","age"]
important_features_smk = ["age","hemoglobin"]

# Boxplot for each target variable
drk_gamma = px.box(df, y=df["gamma_GTP"], x=df["DRK_YN"])
drk_gamma.update_layout(
    title=f'Box Plot of gamma_GTP vs drinking status',
    yaxis_title="gamma_GTP",
    xaxis_title="Drinking Status")

drk_age = px.box(df, y=df["age"], x=df["DRK_YN"])
drk_age.update_layout(
    title=f'Box Plot of age vs drinking status',
    yaxis_title="age",
    xaxis_title="Drinking Status")

smk_age = px.box(df, y=df["age"], x=df["SMK_stat_type_cd"])
smk_age.update_layout(
    title=f'Box Plot of age vs smoking status',
    yaxis_title="age",
    xaxis_title="Smoking Status")

smk_hemog = px.box(df, y=df["hemoglobin"], x=df["SMK_stat_type_cd"])
smk_hemog.update_layout(
    title=f'Box Plot of hemoglobin vs smoking status',
    yaxis_title="hemoglobin",
    xaxis_title="Smoking Status")


# Two tabs
tab1, tab2 = st.tabs(["Drinking Status", "Smoking Status"])

with tab1:
    st.text("Drink Status: 0 (No), 1(Yes)")
    st.plotly_chart(drk_gamma)
    st.plotly_chart(drk_age)

with tab2:
    st.text("Smoke Status: 0 (Never), 1 (Used to smoke but quit), 2 (Still smoke)")
    st.plotly_chart(smk_age)
    st.plotly_chart(smk_hemog)


# add header 2 through markdown
st.markdown("## Target Variable Prediction")

st.markdown('### Data Overview')
st.dataframe(df.head(2))

st.markdown('### Input Parameters')
feature_vec = st.text_input("Input a comma-seperated list of features (20): ", "0,35,81,0.5,0.6,1,1,93,53,85,69,117,30,11.3,1,0.8,20,7,10,23.4")
st.markdown('Example:')
st.markdown('0,45,84,1.2,1.2,1,1,121,80,102,43,133,274,13.4,1,0.7,14,11,16,23.4')
st.markdown('1,40,105.0,1.2,1.2,1.0,1.0,126.0,69.0,125.0,57.0,92.0,83.0,16.4,1.0,1.0,38.0,33.0,21.0,27.8')


x = feature_vec.split(",")
x = [float(i) for i in x]
x = np.array(x).reshape(1, -1)

st.markdown('### Select Model')
model_select = st.radio(
        label="",
        key="visibility",
        options=["Stacked", "Logistic", "GradientBoost", "SVM", "RandomForest","AdaBoost"],
    )

with open (path/f"saved_models/{model_select}PickleDrinking.pkl", 'rb') as file:
    pickle_drink_model = pickle.load(file)
with open (path/f"saved_models/{model_select}PickleSmoking.pkl", 'rb') as file:
    pickle_smoke_model = pickle.load(file)



st.markdown('### Prediction Results')

y_drink_predict = pickle_drink_model.predict(x)[0]
if y_drink_predict==0:
    drink_res='Drinker'
elif y_drink_predict==1:
    drink_res='Non-drinker'
st.markdown("##### Drinking Status: "+drink_res)


y_smoke_predict = pickle_smoke_model.predict(x)[0]
if y_smoke_predict==0:
    smoke_res='Never Smoking'
elif y_smoke_predict==1:
    smoke_res='Used to smoke but quit'
elif y_smoke_predict==2:
    smoke_res="Still smoking"
st.markdown("##### Smoking Status: "+smoke_res)