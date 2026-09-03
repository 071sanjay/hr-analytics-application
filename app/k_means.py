import pandas as pd
import numpy as np
import streamlit as st
from models.model import kmeans_workforce_clusters

import matplotlib.pyplot as plt
import seaborn as sns
import requests

st.header('Employee Workforce Analysis')
st.subheader('Using Kmeans Cluster')

features, scaler, model, clusters, sc = kmeans_workforce_clusters()


st.sidebar.header('Employee Input Features')

EngagementSurvey = st.sidebar.slider(
    "Engagement Survey Score", 1.12, 5.0, 4.28, step=0.01
)

# Employee Satisfaction mapping (1-5 Scale)
emp_sat_dict = {
    1: "1 - Very Dissatisfied",
    2: "2 - Dissatisfied",
    3: "3 - Neutral",
    4: "4 - Satisfied",
    5: "5 - Very Satisfied",
}
EmpSatisfaction = st.sidebar.selectbox(
    "Employee Satisfaction",
    options=list(emp_sat_dict.keys()),
    format_func=lambda x: emp_sat_dict.get(x),
    index=3,  # Default to option 4 ('Satisfied')
)

special_projects_dict = {
    0: "0 Projects",
    1: "1 Project",
    2: "2 Projects",
    3: "3 Projects",
    4: "4 Projects",
    5: "5 Projects",
    6: "6 Projects",
    7: "7 Projects",
    8: "8 Projects",
}
SpecialProjectsCount = st.sidebar.selectbox(
    "Special Projects Count",
    options=list(special_projects_dict.keys()),
    format_func=lambda x: special_projects_dict.get(x),
    index=0,
)

Absences = st.sidebar.slider("Absences (Days)", 1, 20, 10, step=1)

Salary = st.sidebar.slider("Salary ($)", 45046, 250000, 62810, step=1000)

Tenure_years = st.sidebar.slider("Tenure (Years)", 8, 20, 13, step=1)

if st.button('Workforce Analysis'):

    feature_list = list(features.tolist() if hasattr(features, 'tolist') else features)

    input_data = pd.DataFrame([[
            EngagementSurvey, EmpSatisfaction, SpecialProjectsCount, 
            Absences, Salary, Tenure_years
        ]], columns=feature_list)

    # Data scaling
    input_scaler = scaler.transform(input_data)

    # Predict using model
    clusters = model.predict(input_scaler)[0]

    st.success(f"Assigned Cluster: {clusters}")

