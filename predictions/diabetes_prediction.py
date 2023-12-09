import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def predict_diabetes(data):

    data_cleaned = data.drop(
        columns=['discharge_disposition_id', 'patient_id', 'Hospital Name', 'hospital_test_reports', 'readmission'])

    data_heart_failure = data_cleaned[data_cleaned['diabetes'] == 1]

    X = data_heart_failure.drop(columns=['discharge_destination'])
    y = data_heart_failure['discharge_destination']

    X_encoded = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression()
    knn = KNeighborsClassifier()

    ensemble_model = VotingClassifier(estimators=[('logreg', logreg), ('knn', knn)], voting='soft')

    ensemble_model.fit(X_train_scaled, y_train)

    joblib.dump(ensemble_model, 'diabetes2.joblib')

    y_test_pred = ensemble_model.predict(X_test_scaled)

    accuracy_test = accuracy_score(y_test, y_test_pred)
    print(f"Accuracy on the test set (Ensemble): {accuracy_test:.2f}")

    logreg.fit(X_train_scaled, y_train)
    y_test_pred_logreg = logreg.predict(X_test_scaled)
    accuracy_logreg = accuracy_score(y_test, y_test_pred_logreg)
    print(f"Accuracy of Logistic Regression on the test set: {accuracy_logreg:.2f}")

    knn.fit(X_train_scaled, y_train)
    y_test_pred_knn = knn.predict(X_test_scaled)
    accuracy_knn = accuracy_score(y_test, y_test_pred_knn)
    print(f"Accuracy of k-Nearest Neighbors (KNN) on the test set: {accuracy_knn:.2f}")

    hospital_probabilities = ensemble_model.predict_proba(X_test_scaled)

    hospital_ranking = pd.DataFrame(data={'Hospital': df['Hospital Name'].iloc[X_test.index],
                                          'Readmission Probability': hospital_probabilities[:, 1]})
    hospital_ranking = hospital_ranking.groupby('Hospital').mean().sort_values(by='Readmission Probability')

    return (hospital_ranking)
