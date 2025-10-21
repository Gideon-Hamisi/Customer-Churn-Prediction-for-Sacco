import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# --------------------------
# Load Model and Scaler
# --------------------------
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("xgb_churn_model.json")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# --------------------------
# Dashboard Configuration
# --------------------------
st.set_page_config(page_title="Mwalimu Sacco Churn Prediction Dashboard", layout="wide")

st.title("📊 Mwalimu Sacco - Customer Churn Prediction Dashboard")
st.markdown("Predict and analyze member churn risk with **data-driven insights** and interactive ‘What-If’ simulations.")

# --------------------------
# Sidebar: What-If Simulation Inputs
# --------------------------
st.sidebar.header("🎛 What-If Simulation Inputs")

age = st.sidebar.slider("Member Age", 21, 65, 40)
account_age = st.sidebar.slider("Account Age (Months)", 1, 120, 36)
monthly_deposits = st.sidebar.slider("Monthly Deposits (KES)", 1000, 50000, 15000, step=1000)
loan_activity = st.sidebar.selectbox("Has Active Loan?", [0, 1], index=1, format_func=lambda x: "Yes" if x == 1 else "No")
loan_repayment = st.sidebar.slider("Loan Repayment History (%)", 50, 100, 85)
complaints = st.sidebar.slider("Complaints in Last 6 Months", 0, 10, 2)
transactions = st.sidebar.slider("Transactions per Month", 1, 30, 10)
digital_usage = st.sidebar.slider("Digital Usage (%)", 0, 100, 60)

# --------------------------
# Prepare Input Data
# --------------------------
features = [
    "AGE",
    "ACCOUNT_AGE_MONTHS",
    "MONTHLY_DEPOSITS",
    "LOAN_ACTIVITY",
    "LOAN_REPAYMENT_HISTORY",
    "COMPLAINTS_COUNT",
    "TRANSACTION_FREQUENCY",
    "DIGITAL_USAGE"
]

input_data = pd.DataFrame([[
    age, account_age, monthly_deposits, loan_activity,
    loan_repayment, complaints, transactions, digital_usage
]], columns=features)

input_scaled = scaler.transform(input_data)
pred_probs = model.predict_proba(input_scaled)[0]
churn_prob = float(pred_probs[1])
stay_prob = float(pred_probs[0])
prediction = model.predict(input_scaled)[0]

# --------------------------
# Risk Categorization
# --------------------------
if churn_prob < 0.33:
    risk_label, color = "🟢 Low Risk", "green"
elif churn_prob < 0.66:
    risk_label, color = "🟡 Medium Risk", "orange"
else:
    risk_label, color = "🔴 High Risk", "red"

# --------------------------
# Display Results
# --------------------------
st.subheader("🎯 Predicted Member Churn Risk")
st.markdown(
    f"<h2 style='color:{color};'>Churn Probability: {churn_prob*100:.1f}% ({risk_label})</h2>",
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Predicted Status", "Likely to Stay" if prediction == 0 else "Likely to Leave")
with col2:
    st.metric("Churn Probability", f"{churn_prob*100:.1f}%")
with col3:
    st.metric("Retention Probability", f"{stay_prob*100:.1f}%")

# --------------------------
# Feature Importance Plot
# --------------------------
st.markdown("---")
st.subheader("📈 Feature Importance (Model Perspective)")

importances = model.feature_importances_
fig1, ax1 = plt.subplots(figsize=(8, 5), dpi=120)
ax1.barh(features, importances, color="skyblue")
ax1.set_xlabel("Importance Score")
ax1.set_title("Top Features Driving Churn")
st.pyplot(fig1, clear_figure=True)

# --------------------------
# SHAP Explainability
# --------------------------
st.markdown("---")
st.subheader("🔍 Model Explainability (SHAP Analysis)")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_scaled)

fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=120)
shap.summary_plot(shap_values, input_data, plot_type="bar", show=False)
st.pyplot(fig2, clear_figure=True)

# --------------------------
# Interpretation Help
# --------------------------
st.markdown("---")
st.markdown("""
### 🧠 Interpretation:
- **Churn Probability** → Likelihood that a member will leave the Sacco soon.
- **Retention Probability** → Likelihood that a member will stay active.
- **High Churn Probability (≥ 66%)** → Requires immediate engagement.
- Adjust the **sliders** on the left to test how improving deposits, reducing complaints, or increasing digital usage changes churn probability.
""")
