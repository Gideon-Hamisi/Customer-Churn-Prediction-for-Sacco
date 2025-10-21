import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt

# --------------------------
# 1. Generate Synthetic Data
# --------------------------
np.random.seed(42)

n = 1200
df = pd.DataFrame({
    "AGE": np.random.randint(21, 65, n),
    "ACCOUNT_AGE_MONTHS": np.random.randint(1, 120, n),
    "MONTHLY_DEPOSITS": np.random.randint(1000, 50000, n),
    "LOAN_ACTIVITY": np.random.randint(0, 2, n),
    "LOAN_REPAYMENT_HISTORY": np.random.randint(50, 100, n),
    "COMPLAINTS_COUNT": np.random.randint(0, 10, n),
    "TRANSACTION_FREQUENCY": np.random.randint(1, 30, n),
    "DIGITAL_USAGE": np.random.randint(0, 100, n)
})

# Define churn probability using a logical pattern
df["CHURNED"] = (
    (df["COMPLAINTS_COUNT"] > 5).astype(int) |
    (df["DIGITAL_USAGE"] < 20).astype(int) |
    (df["MONTHLY_DEPOSITS"] < 5000).astype(int)
).astype(int)

# --------------------------
# 2. Train-Test Split
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
X = df[features]
y = df["CHURNED"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------
# 3. Feature Scaling
# --------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------
# 4. Train Model (XGBoost)
# --------------------------
model = XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)
model.fit(X_train_scaled, y_train)

# --------------------------
# 5. Evaluate Model
# --------------------------
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("✅ Model Evaluation Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --------------------------
# 6. Save Model and Scaler
# --------------------------
model.save_model("xgb_churn_model.json")
joblib.dump(scaler, "scaler.pkl")

# --------------------------
# 7. Visualize Feature Importance
# --------------------------
fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
importances = model.feature_importances_
ax.barh(features, importances)
ax.set_title("Feature Importance (Mwalimu Sacco Churn Model)")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.show()

print("\n🎯 Model and scaler saved successfully:")
print("   → xgb_churn_model.json")
print("   → scaler.pkl")
