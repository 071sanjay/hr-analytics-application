# import pandas as pd
# import numpy as np

# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
import joblib

# logistic Path
LOGISTIC_MODEL_PATH = 'models/logistic/logistic_model.pkl'
LOGISTIC_SCALER_PATH =  'models/logistic/logistic_scaler.pkl'

def load_logistic_model():
    model = joblib.load(LOGISTIC_MODEL_PATH)
    scaler = joblib.load(LOGISTIC_SCALER_PATH)

    return model, scaler

# df = pd.read_csv('data/HrForMl.csv')
# df['High_abs'] = (df['Absences']>10).astype(int)
# df["EngagementSurvey"] = df["EngagementSurvey"].astype(float)


# def logistic_attrition_predict():
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.metrics import classification_report, confusion_matrix

#     features = ['Salary','EmpSatisfaction','EngagementSurvey', 'Absences',
#     'DaysLateLast30', 'Tenure_years', 'Age', 'SpecialProjectsCount']
#     target = ['Termd']
#     X = df[features]
#     Y = df[target]

#     X_train, X_test, Y_train, Y_test = train_test_split(
#     X, Y, test_size=0.2, random_state=42)

#     scaler = StandardScaler()

#     X_train_scale = scaler.fit_transform(X_train)

#     X_test_scale = scaler.transform(X_test)

#     model = LogisticRegression(
#     solver = 'liblinear',
#     class_weight = 'balanced',
#     random_state = 42
#     )

#     model.fit(X_train_scale, Y_train)

#     joblib.dump(model, LOGISTIC_MODEL_PATH)
#     joblib.dump(scaler, LOGISTIC_SCALER_PATH)

#     return scaler, model