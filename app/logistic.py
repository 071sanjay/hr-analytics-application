import pandas as pd
import streamlit as st
from models.model import logistic_attrition_predict

import matplotlib.pyplot as plt
import seaborn as sns
import requests

st.header('Employee Attrition Analysis')
st.subheader('Using Logistic Regression')

features, scaler, model, Y_pred, cr, cm = logistic_attrition_predict()

API_URL = 'https://hr-analytics-application.onrender.com/predict-retention-logistic'

st.sidebar.header("Employee Input Features")

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

# Days Late in Last 30 Days (Categorical mapping)
days_late_dict = {
    0: "0 days (On Time)",
    1: "1 day late",
    2: "2 days late",
    3: "3 days late",
    4: "4 days late",
    5: "5 days late",
    6: "6 days late",
}

DaysLateLast30 = st.sidebar.selectbox(
    "Days Late (Last 30 Days)",
    options=list(days_late_dict.keys()),
    format_func=lambda x: days_late_dict.get(x),
    index=0,
)

# Special Projects Count mapping
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

# 2. Continuous Numerical Features (Using Sliders with Dataset Limits)

Salary = st.sidebar.slider("Salary ($)", 45046, 250000, 62810, step=1000)

EngagementSurvey = st.sidebar.slider(
    "Engagement Survey Score", 1.12, 5.0, 4.28, step=0.01
)

Absences = st.sidebar.slider("Absences (Days)", 1, 20, 10, step=1)

Tenure_years = st.sidebar.slider("Tenure (Years)", 8, 20, 13, step=1)

Age = st.sidebar.slider("Age", 34, 75, 45, step=1)

if st.button('Predict Attrition'):
    payload = {
        'Salary':Salary,
        'EmpSatisfaction':EmpSatisfaction,
        'EngagementSurvey':EngagementSurvey,
        'Absences':Absences,
        'DaysLateLast30':DaysLateLast30,
        'Tenure_years':Tenure_years,
        'Age':Age, 
        'SpecialProjectsCount':SpecialProjectsCount
    }
    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()

            if result['Prediction Status'] == 0:
                st.write('Likely to return')
                st.success('Employee is unlikely to leave company')

            else:
                st.write('Unlikely to return')
                st.success('Employee is likely to leave company')

        else:
            st.error(f'API Status Code Error: {response.status_code}')
    except requests.exceptions.RequestException as e:
        st.error(f'API Server Error: {e}')


# Visualization
st.subheader('Visualization')

fig, ax = plt.subplots(figsize=(4,4))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels = ["Predicted Retained [0]", "Predicted Terminated [1]"],
            yticklabels = ["Actual Retained [0]", "Actual Terminated [1]"])
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.title("Employee Attrition Confusion Matrix")
st.pyplot(fig)