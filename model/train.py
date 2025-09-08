import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# --- Config ---
# Get project root (ml-ci-cd-pipeline-2/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # this gives .../model
BASE_DIR = os.path.dirname(BASE_DIR)  # go one level up to project root

# Data path (works on Windows + Linux)
data_path = os.path.join(BASE_DIR, "data", "Churn.csv")
print("📂 Loading dataset from:", data_path)

# Load dataset
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Dataset not found at {data_path}")

data = pd.read_csv(data_path)

# ----------------------------
# Example preprocessing: assume 'Churn' is the target
X = data.drop("Churn", axis=1)
y = data["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model inside /model
model_path = os.path.join(BASE_DIR, "model", "churn_model.pkl")
joblib.dump(model, model_path)

print(f"✅ Model trained and saved at: {model_path}")
