import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # go one level up
data_path = os.path.join(BASE_DIR, 'data', 'Churn.csv')

# Load dataset
data = pd.read_csv(data_path)

# Drop customerID column
data = data.drop("customerID", axis=1)

# Convert TotalCharges to numeric
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())

# Load saved encoders
label_encoders = joblib.load(os.path.join(BASE_DIR, "label_encoders.pkl"))
target_encoder = joblib.load(os.path.join(BASE_DIR, "target_encoder.pkl"))

# Apply label encoding to categorical features
for col, le in label_encoders.items():
    if col in data.columns:
        data[col] = le.transform(data[col])

# Features and target
X = data.drop("Churn", axis=1)
y = target_encoder.transform(data["Churn"])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the saved model
model_path = os.path.join(BASE_DIR, "churn_model.pkl")
model = joblib.load(model_path)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")
