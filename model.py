import pandas as pd
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression,LogisticRegression
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def covid19(csv_file_path):
    data = pd.read_csv(csv_file_path)
    label_encoder = LabelEncoder()
    for i in data.columns:
        if (type(data[i].iloc[0]) == str):
            data[i] = label_encoder.fit_transform(data[i])
            data[i] = data[i].astype(float)
    target_column = data['readmission']
    data = data.fillna(0.0).astype(float)
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    mask = ~np.isnan(target_column)
    normalized_data = normalized_data[mask]
    target_column = target_column[mask]
    df=pd.read_csv(csv_file_path)

    model = LinearRegression()
    model.fit(normalized_data, target_column)
    weights = model.coef_
    weights_rounded = np.round(weights, 3)
    data_cleaned = data.dropna()

    X = data_cleaned.drop(columns=['covid19'])
    y = data_cleaned['covid19_test']

    X_encoded = pd.get_dummies(X)

    X_train, X_temp, y_train, y_temp = train_test_split(X_encoded, y, test_size=0.3, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=5)

    knn.fit(X_train, y_train)

    # pickle.dump(knn,open('model.pkl','wb'))

    y_valid_pred = knn.predict(X_valid)
    accuracy_valid = accuracy_score(y_valid, y_valid_pred)

    y_test_pred = knn.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_test_pred)

    data_cleaned = data.drop(columns=['discharge_disposition_id', 'patient_id', 'Hospital Name', 'hospital_test_reports', 'readmission'])

    data_diabetes = data_cleaned[data_cleaned['diabetes'] == 1]

    X = data_diabetes.drop(columns=['discharge_destination'])
    y = data_diabetes['discharge_destination']

    X_encoded = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression()

    logreg.fit(X_train_scaled, y_train)
    joblib.dump(logreg,'diabetes.joblib')

    y_test_pred = logreg.predict(X_test_scaled)

    accuracy_test = accuracy_score(y_test, y_test_pred)
    # print(f"Accuracy on the test set: {accuracy_test:.2f}")

    hospital_probabilities = logreg.predict_proba(X_test_scaled)

    hospital_ranking = pd.DataFrame(data={'Hospital': df['Hospital Name'].iloc[X_test.index], 'Readmission Probability': hospital_probabilities[:, 1]})
    hospital_ranking = hospital_ranking.groupby('Hospital').mean().sort_values(by='Readmission Probability')
    #
    # print("\nRanking of Hospitals based on Readmission Probability for Diabetes:")
    # print(hospital_ranking)
    return hospital_ranking
# a=covid19('datasetnew.csv')
# print(a)