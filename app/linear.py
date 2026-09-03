import pandas as pd
import numpy as np
import streamlit as st
from models.model import linear_salary_predict

import matplotlib.pyplot as plt
import seaborn as sns
import requests

st.header('HR Salary Prediction')
st.subheader('Using linear regression')

features, scaler, model, Y_pred, mae, mse, rmse, r2 = linear_salary_predict()

st.sidebar.header('Employee Input Features')


df = pd.read_csv("data/HrForMl.csv")

features = [
    "GenderID",
    "DeptID",
    "Termd",
    "PositionID",
    "PerfScoreID",
    "EngagementSurvey",
    "EmpSatisfaction",
    "SpecialProjectsCount",
    "Absences",
    "DaysLateLast30",
    "Age",
    "Tenure_years",
]

st.sidebar.header("Feature Inputs")
input_data = {}

# Dynamically generate sliders for all features in a single loop
for feature in features:
    min_val = float(df[feature].min())
    max_val = float(df[feature].max())
    default_val = float(df[feature].median())

    # Set step to 0.01 for floats, 1.0 for integers
    step = 0.01 if df[feature].dtype == "float64" else 1.0

    input_data[feature] = st.sidebar.slider(
        label=feature,
        min_value=min_val,
        max_value=max_val,
        value=default_val,
        step=step,
    )

# Format inputs into DataFrame for model prediction
if st.button('Predict Salary'):
    input_data = pd.DataFrame([
            input_data
    ])[features]


    # Data Scaling
    input_scaler = scaler.transform(input_data)

    # Predict using model
    prediction = model.predict(input_scaler).item()

# Show Answer
    if prediction >= 90000:
        st.success(f"**High Salary Tier:** ${prediction:,.2f}")
        st.caption("This prediction falls in the top 25% of company compensation.")
    elif prediction >= 60000:
        st.info(f"**Mid-Level Salary Tier:** ${prediction:,.2f}")
        st.caption("This prediction falls within the standard average range.")
    else:
        st.warning(f"**Entry / Lower Salary Tier:** ${prediction:,.2f}")
        st.caption(
            "This prediction falls in the lower 25% of company compensation."
        )


# Visualization

st.subheader("Model Performance Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("R² Score", f"{r2:.2f}")
col2.metric("Mean Absolute Error", f"${mae:,.2f}")
col3.metric("Root Mean Sq Error", f"${rmse:,.2f}")