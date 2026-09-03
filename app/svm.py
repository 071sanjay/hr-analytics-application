import pandas as pd
import streamlit as st
from models.model import svm_employee_risk_predict

import matplotlib.pyplot as plt
import seaborn as sns
import requests

st.header('Employee Risk Analysis')
st.subheader('Using SVM')

features, scaler, model, Y_pred, cr, cm = svm_employee_risk_predict()

st.sidebar.header("Employee Input Features")

Salary = st.sidebar.slider("Salary ($)", 45046, 250000, 62810, step=1000)

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

Tenure_years = st.sidebar.slider("Tenure (Years)", 8, 20, 13, step=1)

Age = st.sidebar.slider("Age", 34, 75, 45, step=1)

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

if st.button('Predict Risk'):
    input_data = pd.DataFrame([[
        Salary, EngagementSurvey, EmpSatisfaction, Tenure_years, Age, DaysLateLast30
    ]], columns=features)

    # Data scaling
    input_scaler = scaler.transform(input_data)

    # Predict using model
    prediction = model.predict(input_scaler)[0]

    # Show answer
    if prediction == 0:
        st.write('Low Risk Employee')
    else:
        st.write('High Risk Employee')

# Visualizationi
st.subheader('Visualization')

fig, ax = plt.subplots(figsize=(4,4))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels = ['Predicted Low Risk[0]', 'Predicted High Risk[1]'],
            yticklabels = ['Actual Low Risk[0]', 'Actual High Risk[1]'])
plt.xlabel('Predicted Risk')
plt.ylabel('Actual Risk')
plt.title('SVC: High Risk vs. Low Risk Employees')
st.pyplot(fig)