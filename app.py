from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load(open("models/random_forest_model.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():

    import pandas as pd

features = [float(x) for x in request.form.values()]

feature_names = [
"Time","Amount","V1","V2","V3",
"V4","V5","V6","V7","V8"
]

features_df = pd.DataFrame([features], columns=feature_names)

prediction = model.predict(features_df)

    if prediction[0]==1:
        result="⚠ Fraudulent Transaction"
    else:
        result="✅ Legitimate Transaction"

    return render_template("index.html",prediction_text=result)

if __name__=="__main__":
    app.run(debug=True)