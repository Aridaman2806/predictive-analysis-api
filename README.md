# Predictive Analysis API

This project is a RESTful API for predicting machine downtime or production defects using a manufacturing dataset. The API allows users to upload data, train a machine learning model, and make predictions.

## Deployment
The API is deployed on **Render** and accessible at:

**Base URL:** [https://predictive-analysis-api.onrender.com](https://predictive-analysis-api.onrender.com)

---

## Features
- Upload manufacturing data via CSV.
- Train a Decision Tree model on the uploaded data.
- Make real-time predictions using the trained model.

---

## Endpoints

### 1. Upload Data
**Endpoint:** `/upload`

**Method:** `POST`

**Description:** Upload a CSV file containing manufacturing data.

**Request Example:**
```bash
curl.exe -X POST -F "file=@synthetic_manufacturing_data.csv" https://predictive-analysis-api.onrender.com/upload
```

**Expected Response:**
```json
{
    "message": "File uploaded successfully!"
}
```

---

### 2. Train the Model
**Endpoint:** `/train`

**Method:** `POST`

**Description:** Train the model using the uploaded data.

**Request Example:**
```bash
curl.exe -X POST https://predictive-analysis-api.onrender.com/train
```

**Expected Response:**
```json
{
    "message": "Model trained successfully!",
    "accuracy": 0.85
}
```

---

### 3. Make Predictions
**Endpoint:** `/predict`

**Method:** `POST`

**Description:** Predict downtime based on input features.

**Request Example:**
```bash
curl.exe -X POST -H "Content-Type: application/json" -d "{\"Temperature\": 80, \"Run_Time\": 150}" https://predictive-analysis-api.onrender.com/predict
```

**Expected Response:**
```json
{
    "Downtime": "No",
    "Confidence": 0.92
}
```

---

## Project Setup (For Local Development)

### Prerequisites
- Python 3.x
- Required packages in `requirements.txt`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Aridaman2806/predictive-analysis-api.git
   ```
2. Navigate to the project folder:
   ```bash
   cd predictive-analysis-api
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Flask app:
   ```bash
   python app.py
   ```

The app should now be running on `http://127.0.0.1:5000`.

---

## File Structure
```
 predictive-analysis-api/
 ├── app.py               # Flask API script
 ├── requirements.txt     # Project dependencies
 ├── synthetic_manufacturing_data.csv  # Sample dataset
 ├── README.md            # Documentation
```

---

## Notes
- The API is hosted on Render's free tier, which may result in cold starts (~50 seconds delay after inactivity).
- Update your dataset before retraining for improved accuracy.

---

## Contact
For questions or issues, feel free to reach out via GitHub Issues or email at `aridaman.bhadauria@gmail.com`.

