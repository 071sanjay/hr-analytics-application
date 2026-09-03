import joblib

# logistic Path
LOGISTIC_MODEL_PATH = 'models/logistic/logistic_model.pkl'
LOGISTIC_SCALER_PATH =  'models/logistic/logistic_scaler.pkl'

def load_logistic_model():
    model = joblib.load(LOGISTIC_MODEL_PATH)
    scaler = joblib.load(LOGISTIC_SCALER_PATH)

    return model, scaler