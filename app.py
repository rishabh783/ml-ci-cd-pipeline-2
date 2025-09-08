# app.py
import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# --- Config ---
MODEL_PATH = os.getenv("MODEL_PATH", "model/churn_model.pkl")
ENCODERS_PATH = os.getenv("ENCODERS_PATH", "model/label_encoders.pkl")
TARGET_ENCODER_PATH = os.getenv("TARGET_ENCODER_PATH", "model/target_encoder.pkl")

# --- App ---
app = Flask(__name__)

# Load once at startup
try:
    model = joblib.load(MODEL_PATH)
    label_encoders = joblib.load(ENCODERS_PATH)
    target_encoder = joblib.load(TARGET_ENCODER_PATH)
except Exception as e:
    raise RuntimeError(f"Could not load model/encoders: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}, 200

@app.post("/predict")
def predict():
    """
    Accepts:
    {
      "input": {
          "gender": "Male",
          "SeniorCitizen": 0,
          "Partner": "Yes",
          "Dependents": "No",
          "tenure": 5,
          "PhoneService": "Yes",
          "MultipleLines": "No",
          "InternetService": "DSL",
          "OnlineSecurity": "No",
          "OnlineBackup": "Yes",
          "DeviceProtection": "No",
          "TechSupport": "No",
          "StreamingTV": "Yes",
          "StreamingMovies": "No",
          "Contract": "Month-to-month",
          "PaperlessBilling": "Yes",
          "PaymentMethod": "Electronic check",
          "MonthlyCharges": 70.35,
          "TotalCharges": 350.50
      }
    }
    """
    try:
        payload = request.get_json(force=True)
        features = payload.get("input")
        if features is None:
            return jsonify(error="Missing 'input'"), 400

        # Convert dict → DataFrame
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        elif isinstance(features, list):
            df = pd.DataFrame(features)
        else:
            return jsonify(error="Invalid input format"), 400

        # Preprocess
        for col, le in label_encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col])

        # Ensure numeric conversion for TotalCharges
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

        # Predict
        preds = model.predict(df)
        preds_labels = target_encoder.inverse_transform(preds)

        return jsonify(prediction=preds_labels.tolist()), 200

    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
