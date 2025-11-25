import streamlit as st
import pandas as pd
import joblib

model = joblib.load("loan.pkl")
label_encoders = joblib.load("encoders.pkl")

st.title("💳 Loan Prediction App")

age = st.number_input("Age", 18, 70, 30)
gender = st.selectbox("Gender", label_encoders["gender"].classes_)
occupation = st.selectbox("Occupation", label_encoders["occupation"].classes_)
education = st.selectbox("Education Level", label_encoders["education_level"].classes_)
marital_status = st.selectbox("Marital Status", label_encoders["marital_status"].classes_)
income = st.number_input("Income", 10000, 200000, 50000)
credit_score = st.number_input("Credit Score", 300, 850, 650)

input_data = pd.DataFrame({
    "age": [age],
    "gender": [label_encoders["gender"].transform([gender])[0]],
    "occupation": [label_encoders["occupation"].transform([occupation])[0]],
    "education_level": [label_encoders["education_level"].transform([education])[0]],
    "marital_status": [label_encoders["marital_status"].transform([marital_status])[0]],
    "income": [income],
    "credit_score": [credit_score]
})

if st.button("Predict Loan Status"):
    pred = model.predict(input_data)[0]
    result = label_encoders["loan_status"].inverse_transform([pred])[0]

    if result.lower() == "approved":
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Denied")
