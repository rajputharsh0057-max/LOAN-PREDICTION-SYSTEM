import streamlit as st
import pandas as pd
import joblib

# Load the trained model
preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("model.pkl")

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Loan Approval Prediction System")
st.write("Fill in the applicant details below and click Predict.")

st.markdown("---")

# Input fields

gender = st.selectbox("Gender", ["Male", "Female"])

married = st.selectbox("Married", ["Yes", "No"])

dependents = st.selectbox(
    "Dependents",
    ["0", "1", "2", "3+"]
)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

applicant_income = st.number_input(
    "Applicant Income",
    min_value=0,
    value=5000
)

coapplicant_income = st.number_input(
    "Coapplicant Income",
    min_value=0,
    value=0
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    value=120
)

loan_amount_term = st.number_input(
    "Loan Amount Term",
    min_value=0,
    value=360
)

credit_history = st.selectbox(
    "Credit History",
    [1.0, 0.0]
)

property_area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

if st.button("Predict Loan Approval"):

    input_df = pd.DataFrame({
        "Gender":[gender],
        "Married":[married],
        "Dependents":[dependents],
        "Education":[education],
        "Self_Employed":[self_employed],
        "ApplicantIncome":[applicant_income],
        "CoapplicantIncome":[coapplicant_income],
        "LoanAmount":[loan_amount],
        "Loan_Amount_Term":[loan_amount_term],
        "Credit_History":[credit_history],
        "Property_Area":[property_area]
    })

    input_processed = preprocessor.transform(input_df)
    prediction = model.predict(input_processed)

    if prediction[0] == "Y":
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")