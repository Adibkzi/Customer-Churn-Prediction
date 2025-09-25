from pathlib import Path
import os
import streamlit as st
import pandas as pd
import joblib

# ---------- Page setup ----------
st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="centered")
st.title("📉 Customer Churn Predictor")
st.caption("Enter attributes → get churn probability (RandomForest + preprocessing pipeline)")

# ---------- Robust model loader ----------
# Use __file__ when run via `streamlit run app.py`; fall back to CWD if tested in Jupyter
if "__file__" in globals():
    base_dir = Path(__file__).parent
else:
    base_dir = Path(os.getcwd())

MODEL_PATH = base_dir / "models" / "churn_model.pkl"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            f"❌ Model file not found at: {MODEL_PATH}\n\n"
            "Make sure your notebook saved the trained pipeline with:\n"
            "  joblib.dump(rf, Path('models')/'churn_model.pkl')\n"
            "and that the file is next to app.py inside a 'models/' folder."
        )
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

# ---------- Input form ----------
with st.form("score_form"):
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.number_input("Monthly charges", 15.0, 200.0, 65.0)
    contract = st.selectbox("Contract type", ["month_to_month", "one_year", "two_year"])
    internet = st.selectbox("Internet type", ["dsl", "fiber", "none"])
    pay = st.selectbox("Payment method", ["credit_card", "bank_transfer", "paypal", "mailed_check"])
    region = st.selectbox("Region", ["North", "South", "East", "West"])
    tickets = st.number_input("Support tickets (90d)", 0, 20, 1)
    logins = st.number_input("Logins (30d)", 0, 200, 20)
    late = st.number_input("Late payments (12m)", 0, 3, 0)
    age = st.number_input("Age", 18, 90, 35)
    submitted = st.form_submit_button("Score")

# ---------- Predict ----------
if submitted:
    row = pd.DataFrame([{
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "contract_type": contract,
        "internet_type": internet,
        "payment_method": pay,
        "region": region,
        "support_tickets_90d": tickets,
        "num_logins_30d": logins,
        "late_payments_12m": late,
        "age": age
    }])
    proba = model.predict_proba(row)[0, 1]
    st.metric("Churn probability", f"{proba:.2%}")
    st.info("Tip: Use a top-decile threshold to target the highest-risk customers first.")
