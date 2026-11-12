import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from recap import build_recap, figure_bytes, group_contributions

try:
    import shap
except ImportError:
    shap = None

st.set_page_config(page_title="Telecom churn risk", page_icon="📡")


@st.cache_resource
def load_bundle():
    return joblib.load("telco_model.joblib")


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


bundle = load_bundle()
model = bundle["model"]
columns = bundle["columns"]
defaults = bundle["defaults"]


def set_choice(row, group, choice):
    for col in columns:
        if col.startswith(group + "_"):
            row[col] = 0.0
    key = f"{group}_{choice}"
    if key in row:
        row[key] = 1.0


st.title("Telecom customer churn risk")
st.caption(
    "Demo built on the IBM Telco customer churn dataset (a fictional dataset). "
    "The costs in the sidebar are assumptions you can change, not real data."
)

st.sidebar.header("Business assumptions")
offer_cost = st.sidebar.number_input("Retention offer cost (€)", min_value=1, value=100, step=10)
success_rate = st.sidebar.slider("Offer success rate (%)", 1, 100, 30) / 100
customer_value = st.sidebar.number_input("Value of a retained customer (€, 12 months)", min_value=1, value=800, step=50)

threshold = offer_cost / (success_rate * customer_value)
st.sidebar.metric("Break-even threshold", f"{threshold:.2f}")
if threshold > 1:
    st.sidebar.warning("The offer costs more than it can bring back: no customer is worth contacting.")
else:
    st.sidebar.caption("Contact a customer only if the probability of leaving is above this threshold.")

st.subheader("Customer profile")
col1, col2 = st.columns(2)
with col1:
    contract = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
    tenure = st.slider("Tenure (months)", 4, 72, 12)
    monthly_charge = st.number_input("Monthly charge (€)", min_value=18.0, max_value=125.0, value=70.0, step=1.0)
    internet = st.selectbox("Internet type", ["Cable", "DSL", "Fiber Optic", "No internet"])
    payment = st.selectbox("Payment method", ["Bank Withdrawal", "Credit Card", "Mailed Check"])
with col2:
    age = st.slider("Age", 19, 80, 45)
    dependents = st.slider("Number of dependents", 0, 9, 0)
    referrals = st.slider("Number of referrals", 0, 11, 0)
    gb_download = st.slider("Average monthly download (GB)", 0, 85, 20)
    paperless = st.checkbox("Paperless billing", value=False)

row = dict(defaults)
row["Tenure in Months"] = tenure
row["Monthly Charge"] = monthly_charge
row["Age"] = age
row["Number of Dependents"] = dependents
row["Number of Referrals"] = referrals
row["Referred a Friend"] = 1.0 if referrals > 0 else 0.0
row["Avg Monthly GB Download"] = gb_download
row["Paperless Billing"] = 1.0 if paperless else 0.0
set_choice(row, "Contract", contract)
set_choice(row, "Internet Type", internet)
set_choice(row, "Payment Method", payment)

X_new = pd.DataFrame([row])[columns].astype(float)
proba = float(model.predict_proba(X_new)[0, 1])

st.subheader("Result")
st.metric("Probability of leaving", f"{proba:.0%}")

expected_gain = proba * success_rate * customer_value - offer_cost
if proba >= threshold:
    st.success(f"Contact with a retention offer. Expected gain: {expected_gain:,.0f} €")
else:
    st.info(f"No action. Contacting this customer would cost {-expected_gain:,.0f} € on average.")

st.caption(
    "Limits: only 10 questions are asked. The other variables are set to the typical "
    "value of the training data. The model was trained on about 6,000 customers with "
    "more than 3 months of tenure, so it does not apply to brand-new customers."
)

st.subheader("Recap dashboard")
show_recap = st.toggle("Build a one-page recap for this customer")
if show_recap:
    if shap is None:
        st.warning("The recap needs the shap package, which is not installed here.")
    else:
        explainer = load_explainer(model)
        contributions = group_contributions(columns, explainer.shap_values(X_new))
        profile = [
            ("Contract", contract),
            ("Tenure", f"{tenure} months"),
            ("Monthly charge", f"{monthly_charge:,.0f} €"),
            ("Internet type", internet),
            ("Payment method", payment),
            ("Age", str(age)),
            ("Dependents", str(dependents)),
            ("Referrals", str(referrals)),
            ("Avg monthly download", f"{gb_download} GB"),
            ("Paperless billing", "Yes" if paperless else "No"),
        ]
        fig = build_recap(
            profile=profile,
            proba=proba,
            threshold=threshold,
            expected_gain=expected_gain,
            contact=proba >= threshold,
            assumptions={"offer_cost": offer_cost, "success_rate": success_rate, "customer_value": customer_value},
            contributions=contributions,
        )
        png = figure_bytes(fig, "png")
        pdf = figure_bytes(fig, "pdf")
        plt.close(fig)
        st.image(png)
        left, right = st.columns(2)
        left.download_button("Download the recap (PDF)", pdf, file_name="churn_recap.pdf", mime="application/pdf")
        right.download_button("Download the recap (PNG)", png, file_name="churn_recap.png", mime="image/png")
