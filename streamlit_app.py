import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉")

# Load model + feature list
model = joblib.load("best_churn_model.pkl")
features = joblib.load("churn_model_features.pkl")

st.title("Customer Churn Prediction")
st.write("Enter customer details and predict whether the customer is likely to churn.")

# Inputs
gender_Male = st.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner_Yes = st.selectbox("Partner", ["No", "Yes"])
Dependents_Yes = st.selectbox("Dependents", ["No", "Yes"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
PhoneService_Yes = st.selectbox("Phone Service", ["No", "Yes"])
PaperlessBilling_Yes = st.selectbox("Paperless Billing", ["No", "Yes"])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

MultipleLines_No_phone_service = 0
MultipleLines_Yes = 1 if st.selectbox("Multiple Lines", ["No", "Yes"]) == "Yes" else 0

InternetService_Fiber_optic = 1 if st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"]) == "Fiber optic" else 0
InternetService_No = 1 if st.selectbox("No Internet Service?", ["No", "Yes"]) == "Yes" else 0

OnlineSecurity_Yes = 1 if st.selectbox("Online Security", ["No", "Yes"]) == "Yes" else 0
OnlineBackup_Yes = 1 if st.selectbox("Online Backup", ["No", "Yes"]) == "Yes" else 0
DeviceProtection_Yes = 1 if st.selectbox("Device Protection", ["No", "Yes"]) == "Yes" else 0
TechSupport_Yes = 1 if st.selectbox("Tech Support", ["No", "Yes"]) == "Yes" else 0
StreamingTV_Yes = 1 if st.selectbox("Streaming TV", ["No", "Yes"]) == "Yes" else 0
StreamingMovies_Yes = 1 if st.selectbox("Streaming Movies", ["No", "Yes"]) == "Yes" else 0

Contract_One_year = 1 if st.selectbox("Contract", ["Month-to-month", "One year", "Two year"]) == "One year" else 0
Contract_Two_year = 1 if st.selectbox("Contract (Two year)", ["No", "Yes"]) == "Yes" else 0

PaymentMethod_Credit_card_automatic = 1 if st.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
]) == "Credit card (automatic)" else 0

PaymentMethod_Electronic_check = 1 if st.selectbox("Electronic Check?", ["No", "Yes"]) == "Yes" else 0
PaymentMethod_Mailed_check = 1 if st.selectbox("Mailed Check?", ["No", "Yes"]) == "Yes" else 0

# Build one input row matching training features
input_data = {
    "SeniorCitizen": SeniorCitizen,
    "tenure": tenure,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
    "gender_Male": 1 if gender_Male == "Male" else 0,
    "Partner_Yes": 1 if Partner_Yes == "Yes" else 0,
    "Dependents_Yes": 1 if Dependents_Yes == "Yes" else 0,
    "PhoneService_Yes": 1 if PhoneService_Yes == "Yes" else 0,
    "MultipleLines_No phone service": MultipleLines_No_phone_service,
    "MultipleLines_Yes": MultipleLines_Yes,
    "InternetService_Fiber optic": InternetService_Fiber_optic,
    "InternetService_No": InternetService_No,
    "OnlineSecurity_Yes": OnlineSecurity_Yes,
    "OnlineBackup_Yes": OnlineBackup_Yes,
    "DeviceProtection_Yes": DeviceProtection_Yes,
    "TechSupport_Yes": TechSupport_Yes,
    "StreamingTV_Yes": StreamingTV_Yes,
    "StreamingMovies_Yes": StreamingMovies_Yes,
    "Contract_One year": Contract_One_year,
    "Contract_Two year": Contract_Two_year,
    "PaperlessBilling_Yes": 1 if PaperlessBilling_Yes == "Yes" else 0,
    "PaymentMethod_Credit card (automatic)": PaymentMethod_Credit_card_automatic,
    "PaymentMethod_Electronic check": PaymentMethod_Electronic_check,
    "PaymentMethod_Mailed check": PaymentMethod_Mailed_check,
}

# Create full dataframe with all training columns
row = {col: 0 for col in features}
for k, v in input_data.items():
    if k in row:
        row[k] = v

input_df = pd.DataFrame([row])

if st.button("Predict Churn"):
    prob = model.predict_proba(input_df)[:, 1][0]
    pred = model.predict(input_df)[0]

    st.subheader("Prediction Result")
    st.write(f"**Churn Probability:** {prob:.2%}")
    st.write(f"**Predicted Class:** {'Churn' if pred == 1 else 'Stay'}")

    if pred == 1:
        st.error("This customer is likely to churn.")
    else:
        st.success("This customer is likely to stay.")
