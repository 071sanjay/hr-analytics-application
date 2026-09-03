from fastapi import FastAPI
from app.schema import HrSchema
from app.model import load_logistic_model

app = FastAPI()

model, scaler = load_logistic_model()

# API endpoints/ requests
@app.get('/')
def home():
    return('Welcome to employee attrition prediction')

# post request
import pandas as pd
@app.post('/predict-retention-logistic')
def predict_attrition(data:HrSchema):
    input_data = pd.DataFrame([
        data.model_dump()
    ])
    input_scaler = scaler.transform(input_data)
    prediction = model.predict(input_scaler)[0]
    return {
        'Prediction Status': int(prediction),
        'Status': 'Likely to return' if prediction == 0 else 'Unlikely to return'
    }