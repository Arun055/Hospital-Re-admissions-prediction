
import joblib
import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os
import pymongo

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.template_folder = os.path.abspath("templates")

# Load the data (You need to provide the 'data' DataFrame here)
d = pd.read_csv('datasets/finaldatasets.csv')
data = pd.DataFrame(d)  # Replace this with your actual data

# MongoDB configuration
client = pymongo.MongoClient("mongodb://localhost:27017/")  # Replace the connection URL as needed
db = client["subash"]
usersdb = db['record']
collection_name = "results"

# Sample hospital names
hospital_names = ["Ganga", "Cauvery", "MGM", "Apollo", "Bewell", "Rajiv Gandhi", "Kumaran"]

# Sample location addresses and Google Maps links
location_addresses = {
    "ganga": "https://goo.gl/maps/3gmK1teozAvCLc1J9",
    "cauvery": "https://goo.gl/maps/34q8xwcYiaoTCZW39",
    "mgm": "https://goo.gl/maps/d8bTAswtxXaBbvAZ6",
    "apollo": "https://goo.gl/maps/i6joUSLeE2bfBSZP8",
    "bewell": "https://goo.gl/maps/YBwJboSZRbrZQwNi7",
    "rajiv gandhi": "https://goo.gl/maps/vHEGuvNvedLBQdz19",
    "kumaran": "https://goo.gl/maps/tyGD1X974vkstbxE9"
}

# Function to preprocess data and train the ensemble model
def preprocess_and_train(disease):
    data_cleaned = data.drop(columns=['discharge_disposition_id', 'patient_id', 'Hospital Name', 'hospital_test_reports', 'readmission'])
    data_disease = data_cleaned[data_cleaned[disease] == 1]

    X = data_disease.drop(columns=['discharge_destination'])
    y = data_disease['discharge_destination']

    X_encoded = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, train_size=0.7, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create individual models
    logreg = LogisticRegression()
    knn = KNeighborsClassifier()

    # Create the ensemble model using VotingClassifier
    ensemble_model = VotingClassifier(estimators=[('logreg', logreg), ('knn', knn)], voting='soft')

    # Fit the ensemble model on the training data
    ensemble_model.fit(X_train_scaled, y_train)

    # Save the ensemble model to a file
    joblib.dump(ensemble_model, f'{disease}_ensemble.joblib')

    # Predict using the ensemble model on the test set
    y_test_pred = ensemble_model.predict(X_test_scaled)
    print(y_test_pred)

    # Calculate accuracy of the ensemble model on the test set
    accuracy_test = accuracy_score(y_test, y_test_pred)
    print(f"Accuracy on the test set (Ensemble) for {disease}: {accuracy_test:.2f}")

    # Also, let's print the accuracy of individual models
    logreg.fit(X_train_scaled, y_train)
    y_test_pred_logreg = logreg.predict(X_test_scaled)
    accuracy_logreg = accuracy_score(y_test, y_test_pred_logreg)
    print(f"Accuracy of Logistic Regression on the test set for {disease}: {accuracy_logreg:.2f}")

    knn.fit(X_train_scaled, y_train)
    y_test_pred_knn = knn.predict(X_test_scaled)
    accuracy_knn = accuracy_score(y_test, y_test_pred_knn)
    print(f"Accuracy of k-Nearest Neighbors (KNN) on the test set for {disease}: {accuracy_knn:.2f}")

    hospital_probabilities = ensemble_model.predict_proba(X_test_scaled)

    hospital_probabilities_percentage = (hospital_probabilities[:, 1] * 100).round(2)

    hospital_ranking = pd.DataFrame(data={'Hospital': data['Hospital Name'].iloc[X_test.index],
                                          'Readmission Probability': hospital_probabilities_percentage})
    hospital_ranking = hospital_ranking.groupby('Hospital').mean().sort_values(by='Readmission Probability')

    print("\nRanking of Hospitals")
    print(hospital_ranking)

    # Save the hospital rankings to the MongoDB collection
    save_to_database(hospital_ranking, disease)

# Save the hospital rankings to the MongoDB collection
def save_to_database(hospital_ranking, disease):
    collection = db[collection_name]

    # Delete any existing data for the disease in the collection (for simplicity in this example)
    collection.delete_many({"disease": disease})

    # Save the new data to the collection
    hospital_ranking_records = []
    for index, row in hospital_ranking.iterrows():
        hospital_record = {"hospital_name": index, "readmission_probability": row['Readmission Probability'], "disease": disease}
        hospital_ranking_records.append(hospital_record)

    collection.insert_many(hospital_ranking_records)

# Retrieve the hospital rankings from the MongoDB collection
def retrieve_from_database(disease):
    collection = db[collection_name]
    query = {"disease": disease}
    projection = {"_id": 0, "hospital_name": 1, "readmission_probability": 1}

    data = list(collection.find(query, projection))

    # Convert the retrieved data to a DataFrame
    hospital_ranking = pd.DataFrame(data)

    return hospital_ranking

# Homepage
@app.route('/')
def index():
    return render_template('home.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        reenter_password = request.form['reenter_password']

        # Check if the email already exists in the database (optional)
        if usersdb.find_one({'email': email}):
            return 'Email already registered!'

        # Check if passwords match
        if password != reenter_password:
            return 'Passwords do not match! Please reenter the same password.'

        # Insert the user data into the MongoDB collection
        usersdb.insert_one({'email': email, 'password': password})
        return 'Registration successful!'

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = usersdb.find_one({'email': email, 'password': password})
        if user:
            return render_template("mainhome.html")
        else:
            return 'Login failed. Please check your email and password.'
    return render_template('login.html')

# Prediction and results page
@app.route('/predict', methods=['POST'])
def predict():
    disease = request.form.get('disease')
    # Perform preprocessing and train the ensemble model if not already trained for the disease
    if not os.path.isfile(f'{disease}_ensemble.joblib'):
        preprocess_and_train(disease)

    # Retrieve the hospital rankings from the MongoDB collection
    rankings_from_db = retrieve_from_database(disease)

    # Ensure the DataFrame is not empty before rendering in the template
    if not rankings_from_db.empty:
        # Display the rankings on the console (optional)
        print(f"\nRetrieved Rankings of Hospitals based on Readmission Probability for {disease} (Ensemble Model):")
        print(rankings_from_db)

    return render_template('results.html', ranking=rankings_from_db, hospital_names=hospital_names, location_addresses=location_addresses)


@app.route('/logout', methods=['get'])
def logOut():
    return render_template('home.html')
@app.route('/mainhome', methods=['get'])
def mainhome():
    return render_template('mainhome.html')

@app.route('/predictmain', methods=['get'])
def predictmain():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
