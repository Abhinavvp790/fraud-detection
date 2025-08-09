import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("fraud_model.pkl")

st.set_page_config(page_title="Fraud Detection App", layout="centered")
st.title("💳 Fraud Detection App")
st.write("Enter transaction details to predict if it's fraudulent.")

# Input fields
type_options = ["CASH_OUT", "TRANSFER", "PAYMENT", "DEBIT", "CASH_IN"]
type_input = st.selectbox("Transaction Type", type_options)

amount = st.number_input("Amount", min_value=0.0, step=1.0)
oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, step=1.0)
newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0, step=1.0)
oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0, step=1.0)
newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0, step=1.0)

if st.button("Predict"):
    
    input_data = [[type_input, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]]
    
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    
    st.write(f"**Probability of Fraudulent Transaction:** {prob:.2%}")

    
    st.progress(prob)

    if pred == 1:
        st.error(f"⚠️ This transaction is predicted to be FRAUDULENT! (Probability: {prob:.2%})")
    else:
        st.success(f"✅ This transaction is predicted to be LEGITIMATE. (Probability of fraud: {prob:.2%})")

st.markdown("---")
st.caption("Made with Streamlit | Model: XGBoost Pipeline")