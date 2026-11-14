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


STYLE = """
<style>
.stApp {
    background: linear-gradient(180deg, #ffffff 0%, #eef6fc 45%, #d9eaf7 100%);
    background-attachment: fixed;
}
.stApp {
    isolation: isolate;
}
.stApp::after {
    content: "";
    position: fixed;
    left: 0;
    right: 0;
    bottom: 0;
    height: 130px;
    z-index: -1;
    pointer-events: none;
    background-repeat: repeat-x;
    background-position: 0 100%, 0 100%, 0 100%;
    background-size: 1200px 120px, 900px 90px, 600px 60px;
    background-image:
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1200 120'%3E%3Cpath fill='%23a9d1ee' fill-opacity='0.40' d='M0 60 C200 20 400 100 600 60 C800 20 1000 100 1200 60 V120 H0 Z'/%3E%3C/svg%3E"),
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 900 90'%3E%3Cpath fill='%237fb8e3' fill-opacity='0.28' d='M0 45 C150 75 300 15 450 45 C600 75 750 15 900 45 V90 H0 Z'/%3E%3C/svg%3E"),
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 600 60'%3E%3Cpath fill='%235fa3d8' fill-opacity='0.20' d='M0 30 C100 10 200 50 300 30 C400 10 500 50 600 30 V60 H0 Z'/%3E%3C/svg%3E");
    animation: drift 90s linear infinite;
}
@keyframes drift {
    from { background-position: 0 100%, 0 100%, 0 100%; }
    to { background-position: 1200px 100%, 900px 100%, 600px 100%; }
}
@media (prefers-reduced-motion: reduce) {
    .stApp::after { animation: none; }
}
[data-testid="stHeader"] {
    background: transparent;
}
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.65);
    border-right: 1px solid #d3e5f2;
}
.block-container {
    padding-top: 3rem;
    padding-bottom: 4rem;
}
h1 {
    color: #1f4e79;
    font-weight: 600;
    letter-spacing: -0.01em;
}
h2, h3 {
    color: #2b5f8f;
    font-weight: 500;
}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
    color: #4a6b86;
    opacity: 1;
}
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.85);
    border: 1px solid #d3e5f2;
    border-radius: 16px;
    padding: 14px 18px;
    box-shadow: 0 4px 14px rgba(47, 128, 196, 0.08);
}
[data-testid="stAlert"] {
    border-radius: 14px;
}
.stDownloadButton button, .stButton button {
    border-radius: 999px;
    border: 1px solid #2f80c4;
    color: #2f80c4;
    background: #ffffff;
}
.stDownloadButton button:hover, .stButton button:hover {
    background: #2f80c4;
    color: #ffffff;
    border-color: #2f80c4;
}
</style>
"""

st.markdown(STYLE, unsafe_allow_html=True)


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
    "Will this customer leave? : this demo was built on the IBM Telco customer churn dataset (fictional dataset). "
    "The costs in the sidebar are only assumptions."
)

st.sidebar.header("Business assumptions")
offer_cost = st.sidebar.number_input("Cost of the retention offer (€)", min_value=1, value=100, step=10)
success_rate = st.sidebar.slider("Chance the offer convinces the customer (%)", 1, 100, 30) / 100
customer_value = st.sidebar.number_input("What a kept customer brings in 1 year (€)", min_value=1, value=800, step=50)

threshold = offer_cost / (success_rate * customer_value)
st.sidebar.metric("Minimum risk worth acting on", f"{threshold:.0%}")
if threshold > 1:
    st.sidebar.warning("The offer costs more than it can bring back: no customer is worth contacting.")
else:
    st.sidebar.caption("Below this risk, the offer costs more than it brings back.")

st.subheader("Customer profile")
col1, col2 = st.columns(2)
with col1:
    contract = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
    tenure = st.slider("Months as a customer", 4, 72, 12)
    monthly_charge = st.number_input("Monthly bill (€)", min_value=18.0, max_value=125.0, value=70.0, step=1.0)
    internet = st.selectbox("Internet type", ["Cable", "DSL", "Fiber Optic", "No internet"])
    payment = st.selectbox("Payment method", ["Bank Withdrawal", "Credit Card", "Mailed Check"])
with col2:
    age = st.slider("Age", 19, 80, 45)
    dependents = st.slider("Children / people in the household", 0, 9, 0)
    referrals = st.slider("Number of referrals", 0, 11, 0)
    gb_download = st.slider("Data used per month (GB)", 0, 85, 20)
    paperless = st.checkbox("Receives bills online only", value=False)

row = dict(defaults)
row["Months as a customer"] = tenure
row["Monthly bill"] = monthly_charge
row["Age"] = age
row["Children / people in the household"] = dependents
row["Number of Referrals"] = referrals
row["Referred a Friend"] = 1.0 if referrals > 0 else 0.0
row["Data used per month"] = gb_download
row["Receives bills online only"] = 1.0 if paperless else 0.0
set_choice(row, "Contract", contract)
set_choice(row, "Internet Type", internet)
set_choice(row, "Payment Method", payment)

X_new = pd.DataFrame([row])[columns].astype(float)
proba = float(model.predict_proba(X_new)[0, 1])

st.subheader("Prediction")
st.metric("Chance this customer leaves", f"{proba:.0%}")

expected_gain = proba * success_rate * customer_value - offer_cost
if proba >= threshold:
    st.success(f"Worth contacting: offering the retention deal should bring back about {expected_gain:,.0f} € on average.")
else:
    st.info(f"Not worth contacting: the offer would cost about {-expected_gain:,.0f} € € more than it brings back on average.")

st.caption(
    "Good to know: you only enter 10 details here, and all the others are set to the typical customer. "
    "The model learned from about 6,000 customers who had been with the company for more than 3 months, "
    "so it is not meant for brand-new customers."
)

st.subheader("Recap dashboard")
show_recap = st.toggle("Create a one-page summary for this customer")
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
