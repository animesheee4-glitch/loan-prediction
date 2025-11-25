import streamlit as st
import pandas as pd
import pickle

# ---------------------------
# LOAD MODEL & LABEL ENCODERS
# ---------------------------
with open("loan.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("💳 Loan Prediction App")
st.write("This app predicts loan approval using a pre-trained ML model.")

st.header("Enter Applicant Details")

age = st.number_input("Age", min_value=18, max_value=70, value=30)
gender = st.selectbox("Gender", label_encoders["gender"].classes_)
occupation = st.selectbox("Occupation", label_encoders["occupation"].classes_)
education = st.selectbox("Education Level", label_encoders["education_level"].classes_)
marital_status = st.selectbox("Marital Status", label_encoders["marital_status"].classes_)
income = st.number_input("Annual Income", min_value=10000, max_value=200000, value=50000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)

# ---------------------------
# ENCODE USER INPUT
# ---------------------------
input_data = pd.DataFrame({
    "age": [age],
    "gender": [label_encoders["gender"].transform([gender])[0]],
    "occupation": [label_encoders["occupation"].transform([occupation])[0]],
    "education_level": [label_encoders["education_level"].transform([education])[0]],
    "marital_status": [label_encoders["marital_status"].transform([marital_status])[0]],
    "income": [income],
    "credit_score": [credit_score]
})

# ---------------------------
# PREDICT
# ---------------------------
if st.button("Predict Loan Status"):
    prediction = model.predict(input_data)[0]
    result = label_encoders["loan_status"].inverse_transform([prediction])[0]

    if result.lower() == "approved":
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Denied")
