import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from streamlit_extras.add_vertical_space import add_vertical_space
from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode

from pathlib import Path
path = Path(__file__).parent


# Read data
# df = pd.read_csv("./datasets/downsampled_dataset_after_feature_selection.csv")
df = pd.read_csv(path/"datasets/downsampled_dataset_after_feature_selection.csv")


## Streamlit Setup
# set title and page width
st.set_page_config(page_title='Smoking and Drinking Status Prediction App', layout="wide")
st.title('Smoking and Drinking Status Prediction App')

# title for boxplot
st.markdown('## Boxplots for Important Features with Drinking and Smoking Status')

# Important features selected
important_features_drk = ["gamma_GTP","age"]
important_features_smk = ["age","hemoglobin"]

# Boxplot for each target variable
## For Drinking status
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

## For Somking Status
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


# Two tabs for drinking and somking
tab1, tab2 = st.tabs(["Drinking Status", "Smoking Status"])

with tab1:
    # description of target variable: drinking
    st.text("Drink Status: 0 (No), 1(Yes)")
    st.write("")
    # add columns to put plots in one row
    row1_space1, row1_1, row1_space2, row1_2, row1_space3 = st.columns(
        (0.1, 1, 0.1, 1, 0.1)
    )
    with row1_1:
        st.plotly_chart(drk_gamma,use_container_width=True)
    with row1_2:
        st.plotly_chart(drk_age,use_container_width=True)

with tab2:
    # description of target variable: smoking
    st.text("Smoke Status: 0 (Never), 1 (Used to smoke but quit), 2 (Still smoke)")
    add_vertical_space()
    # add columns to put plots in one row
    row2_space1, row2_1, row2_space2, row2_2, row2_space3 = st.columns(
        (0.1, 1, 0.1, 1, 0.1)
    )
    with row2_1:
        st.plotly_chart(smk_age,use_container_width=True)
    with row2_2:
        st.plotly_chart(smk_hemog,use_container_width=True)



st.markdown("## Target Variable Prediction")

# display first 10 data
st.markdown('### Data Overview')
AgGrid(
    df.head(10), columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS
)

# set radio to allow user to select different model
st.markdown('### Select Model')
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
model_select = st.radio(
        label="Select Model",
        key="models",
        options=["Stacked", "Logistic", "GradientBoost", "SVM", "RandomForest","AdaBoost"],
        label_visibility="collapsed"
    )

st.markdown('### Input Parameters')
# set columns to align data input with a button to click to predict
col1, col2 = st.columns([3, 1])
with col1:
    feature_vec = st.text_input(label="Input a comma-seperated list of features (20): ", 
        value="0,35,81,0.5,0.6,1,1,93,53,85,69,117,30,11.3,1,0.8,20,7,10,23.4")
with col2:
    # in order to align button with text_input
    st.markdown('##')
    # if click button, then predict new data based on input text, else use the default example
    predict = st.button('Predict')


if model_select or predict:
    x = feature_vec.split(",")
    x = [float(i) for i in x]
    x = np.array(x).reshape(1, -1)
else:
    x=np.array([0,35,81,0.5,0.6,1,1,93,53,85,69,117,30,11.3,1,0.8,20,7,10,23.4]).reshape(1,-1)

# if predict:
#     x = feature_vec.split(",")
#     x = [float(i) for i in x]
#     x = np.array(x).reshape(1, -1)
# else:
#     x=np.array([0,35,81,0.5,0.6,1,1,93,53,85,69,117,30,11.3,1,0.8,20,7,10,23.4]).reshape(1,-1)


# 5 columns to display 5 example, allow to copy texts
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
with col1:
    st.info('''Example 1: \n
    0,45,84,
    1.2,1.2,1,
    1,121,80,
    102,43,133,
    274,13.4,1,
    0.7,14,11,
    16,23.4
    '''
    )
with col2:
    st.info('''Example 2: \n
    1,40,105.0,
    1.2,1.2,1.0,
    1.0,126,69,
    125,57,92,
    83,16.4,1,
    1,38.0,33,
    21,27.8
    ''')
with col3:
    st.info('''Example 3: \n
    0,75,90,
    0.6,0.7,1,
    1,140,80,
    94,40,56,
    165,13,1,
    0.5,20,24,
    30,23.8
    ''')
with col4:
    st.info('''Example 4: \n
    1,40,100,
    1.2,1.5,1,
    1,160,110,
    100,42,128,
    189,16.6,1,
    0.7,24,40,
    45,29.4
    ''')
with col5:
    st.info('''Example 5: \n
    1,30,81,
    1,0.8,1,
    1,116,77,
    89,51,126,
    84,15.5,1,
    0.9,55,85,
    47,24.2
    ''')

# based on selected model, read corresponding model pickle file
# with open (f"./saved_models/{model_select}PickleDrinking.pkl", 'rb') as file:
with open (path/f"saved_models/{model_select}PickleDrinking.pkl", 'rb') as file:
    pickle_drink_model = pickle.load(file)
# with open (f"./saved_models/{model_select}PickleSmoking.pkl", 'rb') as file:
with open (path/f"saved_models/{model_select}PickleSmoking.pkl", 'rb') as file:
    pickle_smoke_model = pickle.load(file)


# prediction results with custom styling
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-weight: bold;
}
.result-text {
    font-weight: bold;
}
.drinker {
    color: pink;
}
.non-drinker {
    color: green;
}
.never-smoking {
    color: green;
}
.used-to-smoke {
    color: orange;
}
.still-smoking {
    color: red;
}
</style>
""", unsafe_allow_html=True)



# Drinking prediction
y_drink_predict = pickle_drink_model.predict(x)[0]
if y_drink_predict == 0:
    drink_res = 'Drinker'
    drink_icon = "🍷"
    drink_class = "drinker"
elif y_drink_predict == 1:
    drink_res = 'Non-drinker'
    drink_icon = "🚱"
    drink_class = "non-drinker"

# Smoking prediction
y_smoke_predict = pickle_smoke_model.predict(x)[0]
if y_smoke_predict == 0:
    smoke_res = 'Never Smoking'
    smoke_icon = "🚭"
    smoke_class = "never-smoking"
elif y_smoke_predict == 1:
    smoke_res = 'Used to smoke but quit'
    smoke_icon = "🚬"
    smoke_class = "used-to-smoke"
elif y_smoke_predict == 2:
    smoke_res = "Still smoking"
    smoke_icon = "🔥"
    smoke_class = "still-smoking"

# Display results
st.markdown(f'<p>Drinking Status: <span class="result-text {drink_class}">{drink_icon} {drink_res}</span></p >', unsafe_allow_html=True)
st.markdown(f'<p>Smoking Status: <span class="result-text {smoke_class}">{smoke_icon} {smoke_res}</span></p >', unsafe_allow_html=True)
