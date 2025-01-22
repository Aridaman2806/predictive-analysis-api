from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import os
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Predictive Analysis API!"})

# Upload endpoint
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    file.save("uploaded_data.csv")
    logger.info("File uploaded successfully!")
    return jsonify({"message": "File uploaded successfully!"})

# Train endpoint
@app.route('/train', methods=['POST'])
def train_model():
    if not os.path.exists("uploaded_data.csv"):
        return jsonify({"error": "No data found. Please upload a file first."}), 400

    data = pd.read_csv("uploaded_data.csv")
    X = data[['Temperature', 'Run_Time']]
    y = data['Downtime_Flag']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    joblib.dump(model, 'model.pkl')
    logger.info(f"Model trained with accuracy: {accuracy}")
    return jsonify({"message": "Model trained successfully!", "accuracy": accuracy})

# Predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    model = joblib.load('model.pkl')
    X_new = [[input_data['Temperature'], input_data['Run_Time']]]

    prediction = model.predict(X_new)
    confidence = model.predict_proba(X_new).max()

    result = {"Downtime": "Yes" if prediction[0] == 1 else "No", "Confidence": round(confidence, 2)}
    logger.info(f"Prediction made: {result}")
    return jsonify(result)

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"An error occurred: {str(e)}")
    return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
