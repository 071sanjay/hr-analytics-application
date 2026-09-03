# HR Analytics — Employee Attrition Predictor

Predict whether an employee is likely to leave a company using HR metrics like salary, job satisfaction, engagement score, absences, and tenure.

This end-to-end machine learning project covers data exploration, model training, a FastAPI backend, and an interactive Streamlit UI.

---

## Live Links

* **Interactive App:** [Streamlit Dashboard](https://hranalyticsapplication-frontend.streamlit.app/)
* **API Documentation:** [FastAPI Docs](https://hr-analytics-application.onrender.com/docs)

---

## Overview & Architecture

* **Data & Modeling:** Performed EDA and model training in Jupyter Notebooks across classification, regression, and clustering algorithms.
* **Backend:** FastAPI service that exposes the trained Logistic Regression model for real-time attrition predictions.
* **Frontend:** Multi-page Streamlit dashboard for interactive user inputs and visual analysis.

---

## Models & Functionality

| Model | Task | Deployment |
| --- | --- | --- |
| **Logistic Regression** | Attrition prediction (Yes/No) | Wired to live FastAPI backend |
| **Support Vector Machine (SVM)** | Attrition prediction (Yes/No) | Executed locally in app |
| **K-Means Clustering** | Employee segmentation | Executed locally in app |
| **Linear Regression** | Salary prediction | Executed locally in app |

---

## Tech Stack

* **Language:** Python
* **Data Science & ML:** pandas, scikit-learn
* **API Framework:** FastAPI
* **Frontend:** Streamlit
* **Deployment:** Render (API), Streamlit Community Cloud (Frontend)

---

## Local Setup

```bash
# 1. Backend Service
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

# 2. Frontend Application (in a new terminal)
cd frontend
pip install -r requirements.txt
streamlit run home.py
```
