import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("loan.csv")

# Encode categorical variables
categorical_cols = ["gender", "occupation", "education_level", "marital_status", "loan_status"]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Streamlit UI
st.title("💳 Loan Prediction App")
st.write(f"Model Accuracy: {acc*100:.2f}%")

st.header("Enter Applicant Details")

age = st.number_input("Age", min_value=18, max_value=70, value=30)
gender = st.selectbox("Gender", label_encoders["gender"].classes_)
occupation = st.selectbox("Occupation", label_encoders["occupation"].classes_)
education = st.selectbox("Education Level", label_encoders["education_level"].classes_)
marital_status = st.selectbox("Marital Status", label_encoders["marital_status"].classes_)
income = st.number_input("Annual Income", min_value=10000, max_value=200000, value=50000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)

# Encode inputs
input_data = pd.DataFrame({
    "age": [age],
    "gender": [label_encoders["gender"].transform([gender])[0]],
    "occupation": [label_encoders["occupation"].transform([occupation])[0]],
    "education_level": [label_encoders["education_level"].transform([education])[0]],
    "marital_status": [label_encoders["marital_status"].transform([marital_status])[0]],
    "income": [income],
    "credit_score": [credit_score]
})

# Prediction
if st.button("Predict Loan Status"):
    prediction = model.predict(input_data)[0]
    result = label_encoders["loan_status"].inverse_transform([prediction])[0]
    if result == "Approved":
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Denied")