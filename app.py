from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("models/random_forest_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get form values
    features = [float(x) for x in request.form.values()]

    # Feature names used during training
    feature_names = [
        "Time", "Amount", "V1", "V2", "V3",
        "V4", "V5", "V6", "V7", "V8"
    ]

    # Convert input to dataframe
    features_df = pd.DataFrame([features], columns=feature_names)

    # Prediction
    prediction = model.predict(features_df)

    if prediction[0] == 1:
        result = "⚠ Fraudulent Transaction"
    else:
        result = "✅ Legitimate Transaction"

    return render_template("index.html", prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)