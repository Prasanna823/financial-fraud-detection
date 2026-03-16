import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/random_forest_model.pkl")

st.title("Financial Transaction Fraud Detection")

st.write("Enter transaction details to predict fraud.")

# Inputs
time = st.number_input("Transaction Time", min_value=0.0)
amount = st.number_input("Transaction Amount", min_value=0.0)
v1 = st.number_input("Transaction Frequency Behavior")

device = st.selectbox(
"Device/Login Pattern",
["Mobile","Laptop","Desktop","Unknown Device"]
)

v3 = st.number_input("Spending Pattern Deviation")

location = st.selectbox(
"Transaction Location Pattern",
["Home","Known","New","International"]
)

activity = st.selectbox(
"Account Activity Behavior",
["Normal","High","Unusual"]
)

payment = st.selectbox(
"Payment Method",
["Card","Cash","Online","UPI","Wallet"]
)

v7 = st.number_input("Merchant Interaction Behavior")
v8 = st.number_input("Overall Risk Score Pattern")

# Convert categorical values
device_map={"Mobile":0,"Laptop":1,"Desktop":2,"Unknown Device":3}
location_map={"Home":0,"Known":1,"New":2,"International":3}
activity_map={"Normal":0,"High":1,"Unusual":2}
payment_map={"Card":0,"Cash":1,"Online":2,"UPI":3,"Wallet":4}

device_val=device_map[device]
location_val=location_map[location]
activity_val=activity_map[activity]
payment_val=payment_map[payment]

if st.button("Predict Fraud"):

    features=[
        time,amount,v1,device_val,v3,
        location_val,activity_val,payment_val,v7,v8
    ]

    feature_names=[
        "Time","Amount","V1","V2","V3",
        "V4","V5","V6","V7","V8"
    ]

    df=pd.DataFrame([features],columns=feature_names)

    prediction=model.predict(df)

    if prediction[0]==1:
        st.error("⚠ Fraudulent Transaction")
    else:
        st.success("✅ Legitimate Transaction")