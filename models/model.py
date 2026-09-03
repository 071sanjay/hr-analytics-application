import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('data/HrForMl.csv')
df['High_abs'] = (df['Absences']>10).astype(int)
df["EngagementSurvey"] = df["EngagementSurvey"].astype(float)

def linear_salary_predict():    
    features =['GenderID', 'DeptID', 'Termd', 'PositionID', 'PerfScoreID', 'EngagementSurvey','EmpSatisfaction',
            'SpecialProjectsCount', 'Absences', 'DaysLateLast30', 'Age', 'Tenure_years']
    target = ['Salary']

    X = df[features]
    Y = df[target]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size = 0.2, random_state = 42)

    scaler = StandardScaler()

    X_train_scale = scaler.fit_transform(X_train)
    X_test_scale = scaler.transform(X_test)

    model = LinearRegression()

    model.fit(X_train_scale, Y_train)

    Y_pred = model.predict(X_test_scale)

    mae = mean_absolute_error(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, Y_pred)

    return features, scaler, model, Y_pred, mae, mse, rmse, r2


def logistic_attrition_predict():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix

    features = ['Salary','EmpSatisfaction','EngagementSurvey', 'Absences',
    'DaysLateLast30', 'Tenure_years', 'Age', 'SpecialProjectsCount']
    target = ['Termd']
    X = df[features]
    Y = df[target]

    X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

    scaler = StandardScaler()

    X_train_scale = scaler.fit_transform(X_train)

    X_test_scale = scaler.transform(X_test)

    model = LogisticRegression(
    solver = 'liblinear',
    class_weight = 'balanced',
    random_state = 42
    )

    model.fit(X_train_scale, Y_train)

    Y_pred = model.predict(X_test_scale)

    cr = classification_report(Y_test, Y_pred)
    cm = confusion_matrix(Y_test, Y_pred)

    return features, scaler, model, Y_pred, cr, cm

def svm_employee_risk_predict():
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report, confusion_matrix

    features = ['Salary', 'EngagementSurvey', 'EmpSatisfaction', 'Tenure_years', 'Age', 'DaysLateLast30']
    target = ['High_abs']

    df_sample = df.sample(n=300, random_state=42)

    X = df_sample[features]
    Y = df_sample['High_abs']

    X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()

    X_train_scale = scaler.fit_transform(X_train)
    X_test_scale = scaler.transform(X_test)

    model = SVC(
    kernel='linear',
    C = 0.1,
    gamma = 0.1
    )

    model.fit(X_train_scale, Y_train)

    Y_pred = model.predict(X_test_scale)

    cr = classification_report(Y_test, Y_pred)
    cm = confusion_matrix(Y_test, Y_pred)

    return features, scaler, model, Y_pred, cr, cm


def kmeans_workforce_clusters():
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import warnings
    warnings.filterwarnings('ignore')

    features = ['EngagementSurvey', 'EmpSatisfaction', 'SpecialProjectsCount', 'Absences', 'Salary', 'Tenure_years']

    X = df[features]
    scaler = StandardScaler()

    X_scale = scaler.fit_transform(X)

    k = 3

    model = KMeans(n_clusters=k)
    clusters = model.fit_predict(X_scale)

    df['clusters'] = clusters

    sc = silhouette_score(X_scale, labels=clusters)

    return X, scaler, model, clusters, sc







