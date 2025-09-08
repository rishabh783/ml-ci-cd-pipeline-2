import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load the dataset
data_path = os.path.join(os.path.dirname(__file__), "data", "Churn.csv")
data = pd.read_csv(data_path)

# Drop customerID column
data = data.drop("customerID", axis=1)

# Convert TotalCharges to numeric (some are blank strings)
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())

# Encode categorical variables
label_encoders = {}
for col in data.select_dtypes(include=["object"]).columns:
    if col != "Churn":  # Don't encode target yet
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

# Features and target
X = data.drop("Churn", axis=1)
y = data["Churn"]

# Encode target (Yes=1, No=0)
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "churn_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")  # save encoders for preprocessing
joblib.dump(target_encoder, "target_encoder.pkl")
