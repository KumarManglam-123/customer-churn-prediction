import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

st.title("Customer Churn Prediction App")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])

if st.button("Predict Churn"):

    input_dict = {
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Contract": contract,
        "InternetService": internet,
        "gender": gender,
        "SeniorCitizen": senior,
    }

    input_df = pd.DataFrame([input_dict])

    # One-hot encode like training
    input_df = pd.get_dummies(input_df)

    # Add missing columns
    for col in columns:
        if col not in input_df:
            input_df[col] = 0

    # Ensure same column order
    input_df = input_df[columns]

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("Customer is likely to churn ❌")
    else:
        st.success("Customer will stay ✅")
