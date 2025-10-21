import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ========== Load model and scaler ==========
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit page setup
st.set_page_config(page_title="Mwalimu Sacco Churn Prediction", layout="wide")
st.title("📊 Mwalimu Sacco – Member Churn Prediction Dashboard")
st.markdown("""
Use this dashboard to predict which members are at risk of leaving or reducing engagement.
Upload a CSV file **or** use the simulation sliders below.
""")

# ========== Sidebar: File Upload ==========
st.sidebar.header("📂 Upload Member Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# Expected feature columns
expected_cols = [
    'AGE', 'GENDER', 'ACCOUNT_AGE_MONTHS', 'MONTHLY_DEPOSITS',
    'LOAN_ACTIVITY', 'LOAN_REPAYMENT_HISTORY', 'COMPLAINTS_COUNT',
    'TRANSACTION_FREQUENCY', 'DIGITAL_USAGE'
]

# ========== If user uploads data ==========
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview", df.head())

    # Scale numeric values
    X_scaled = scaler.transform(df)

    # Predict probabilities
    churn_probs = model.predict_proba(df)[:, 1]
    df['CHURN_PROBABILITY'] = np.round(churn_probs, 3)

    # Assign risk category
    df['RISK_LEVEL'] = pd.cut(
        df['CHURN_PROBABILITY'],
        bins=[0, 0.33, 0.66, 1],
        labels=['Low', 'Medium', 'High']
    )

    st.write("### Predictions", df[['CHURN_PROBABILITY', 'RISK_LEVEL']].head())

    # Show summary
    risk_counts = df['RISK_LEVEL'].value_counts()
    st.bar_chart(risk_counts)

else:
    st.info("Upload a CSV file or use the sliders below to simulate predictions.")

# ========== Interactive Simulation ==========
st.markdown("---")
st.header("🎛️ What-If Simulation")

# Collect input features
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 21, 65, 40)
    gender = st.selectbox("Gender", ["Male", "Female"])
    account_age = st.slider("Account Age (months)", 6, 180, 36)

with col2:
    deposits = st.slider("Average Monthly Deposits (KES)", 2000, 50000, 20000, step=1000)
    loan_activity = st.selectbox("Has Active Loan?", ["Yes", "No"])
    loan_history = st.slider("Loan Repayment History (%)", 50, 100, 85)

with col3:
    complaints = st.slider("Complaints (last 6 months)", 0, 10, 1)
    transactions = st.slider("Monthly Transactions", 1, 20, 8)
    digital = st.slider("Digital Usage (%)", 0, 100, 60)

# Prepare single record
input_data = pd.DataFrame({
    'AGE': [age],
    'GENDER': [1 if gender == "Male" else 0],
    'ACCOUNT_AGE_MONTHS': [account_age],
    'MONTHLY_DEPOSITS': [deposits],
    'LOAN_ACTIVITY': [1 if loan_activity == "Yes" else 0],
    'LOAN_REPAYMENT_HISTORY': [loan_history],
    'COMPLAINTS_COUNT': [complaints],
    'TRANSACTION_FREQUENCY': [transactions],
    'DIGITAL_USAGE': [digital]
})

# Predict
prob = model.predict_proba(input_data)[:, 1][0]
risk = (
    "High" if prob > 0.66 else
    "Medium" if prob > 0.33 else
    "Low"
)

# Display result
st.markdown(f"### 🧾 Predicted Churn Probability: **{prob:.2f}** ({risk} Risk)")

# ========== SHAP Explainability ==========
st.markdown("---")
st.header("🔍 Model Explainability (SHAP)")

# Compute SHAP values for single prediction
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_data)

# Create SHAP plot
fig, ax = plt.subplots()
shap.summary_plot(shap_values, input_data, plot_type="bar", show=False)
st.pyplot(fig)

st.caption("SHAP shows which features most influenced this member's churn prediction.")
