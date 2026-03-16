import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/random_forest_model.pkl")

st.title("Financial Transaction Fraud Detection")

time = st.number_input("Transaction Time")
amount = st.number_input("Transaction Amount")
freq = st.number_input("Transaction Frequency Behavior")

device = st.selectbox(
"Device/Login Pattern",
["Mobile","Laptop","Desktop","Unknown Device"]
)

spending = st.number_input("Spending Pattern Deviation")

location = st.selectbox(
"Location Pattern",
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

merchant = st.number_input("Merchant Interaction Behavior")
risk = st.number_input("Overall Risk Score")

device_map={"Mobile":0,"Laptop":1,"Desktop":2,"Unknown Device":3}
location_map={"Home":0,"Known":1,"New":2,"International":3}
activity_map={"Normal":0,"High":1,"Unusual":2}
payment_map={"Card":0,"Cash":1,"Online":2,"UPI":3,"Wallet":4}

if st.button("Predict Fraud"):

    data={
    "Time":[time],
    "Amount":[amount],
    "Transaction_Frequency_Behavior":[freq],
    "Device_Login_Pattern":[device_map[device]],
    "Spending_Pattern_Deviation":[spending],
    "Location_Pattern":[location_map[location]],
    "Account_Activity_Behavior":[activity_map[activity]],
    "Payment_Method":[payment_map[payment]],
    "Merchant_Interaction_Behavior":[merchant],
    "Overall_Risk_Score":[risk]
    }

    df=pd.DataFrame(data)

    prediction=model.predict(df)

    if prediction[0]==1:
        st.error("⚠ Fraudulent Transaction")
    else:
        st.success("✅ Legitimate Transaction")