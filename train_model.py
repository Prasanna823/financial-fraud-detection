import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("dataset/fraud_dataset.csv")

# Convert categorical values to numbers

data["Device_Login_Pattern"] = data["Device_Login_Pattern"].map({
    "Mobile":0,
    "Laptop":1,
    "Desktop":2,
    "Unknown Device":3
})

data["Location_Pattern"] = data["Location_Pattern"].map({
    "Home":0,
    "Known":1,
    "New":2,
    "International":3
})

data["Account_Activity_Behavior"] = data["Account_Activity_Behavior"].map({
    "Normal":0,
    "High":1,
    "Unusual":2
})

data["Payment_Method"] = data["Payment_Method"].map({
    "Card":0,
    "Cash":1,
    "Online":2,
    "UPI":3,
    "Wallet":4
})

# Split features and target
X = data.drop("Class", axis=1)
y = data["Class"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/random_forest_model.pkl")

print("Model trained successfully!")