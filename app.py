# ------------------------------------------------------------
# AEGIS Telecom Churn Risk Model — Developed by T. Divyasri
# FINAL VERSION — CSV RESTORED + FULL PIPELINE
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    precision_recall_curve, confusion_matrix
)

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="AEGIS Telecom Churn Risk Model", layout="wide")

# ------------------------------------------------------------
# GLOBAL STYLE
# ------------------------------------------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(140deg,#E24587,#C7369B,#8C2CAC,#6021A7);
    padding: 20px;
}
.banner {
    background: linear-gradient(90deg,#00C96B,#00AFC4,#008DE5,#0066FF);
    padding: 22px;
    border-radius: 16px;
    text-align: center;
    color: white;
    font-size: 30px;
    font-weight: 800;
    margin-bottom: 30px;
}
.section-title {
    color: white;
    font-size: 22px;
    font-weight: 800;
    margin-top: 25px;
}
.metric-card {
    background: linear-gradient(135deg,#00C96B,#00B6BD,#008DE5);
    padding: 18px;
    border-radius: 16px;
    text-align: center;
    color: white;
    font-size: 22px;
    font-weight: 800;
}
.metric-label { font-size:18px; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.markdown("""
<div class="banner">
AEGIS Telecom Churn Risk Model — Developed by T. Divyasri
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# SIDEBAR CONTROLS
# ------------------------------------------------------------
st.sidebar.header("⚙ Model Controls")
folds = st.sidebar.slider("CV Folds", 3, 10, 5)
lr = st.sidebar.number_input("Learning Rate", 0.001, 1.0, 0.05, 0.01)
epochs = st.sidebar.slider("Epochs", 300, 1500, 600, 50)
l2 = st.sidebar.number_input("L2 Regularization", 0.0, 0.1, 1e-4, format="%.5f")

# ------------------------------------------------------------
# DATA UPLOAD
# ------------------------------------------------------------
uploaded_file = st.file_uploader("Upload Telco Churn CSV", type=["csv"])
if not uploaded_file:
    st.stop()
df = pd.read_csv(uploaded_file)

# ------------------------------------------------------------
# MODEL HELPERS
# ------------------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_model(X, y, lr, epochs, l2):
    w = np.zeros(X.shape[1])
    b = 0
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        err = p - y
        w -= lr * ((X.T @ err) / len(y) + l2 * w)
        b -= lr * err.mean()
    return w, b

def build_signals(df):
    df["Churn"] = df["Churn"].map({"Yes":1,"No":0})
    for c in ["tenure","MonthlyCharges","TotalCharges"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    URS = ((df["tenure"]<=12) + 
           (df["MonthlyCharges"]>=80) + 
           (df["InternetService"]=="Fiber optic")).astype(int)

    BRS = ((df["PaperlessBilling"]=="Yes") +
           (df["PaymentMethod"]=="Electronic check")).astype(int)

    SRS = ((df["OnlineSecurity"]=="No") +
           (df["TechSupport"]=="No")).astype(int)

    CRS = df["Contract"].map({
        "Month-to-month":2,
        "One year":1,
        "Two year":0
    }).astype(int)

    X = np.log1p(np.c_[URS, BRS, SRS, CRS])
    return X, df["Churn"].values, URS, BRS, SRS, CRS

X_raw, y, URS, BRS, SRS, CRS = build_signals(df)

# ------------------------------------------------------------
# CROSS-VALIDATION TRAINING
# ------------------------------------------------------------
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
oof = np.zeros_like(y, float)

for tr, te in skf.split(X_raw, y):
    scaler = StandardScaler().fit(X_raw[tr])
    Xtr, Xte = scaler.transform(X_raw[tr]), scaler.transform(X_raw[te])
    w, b = train_model(Xtr, y[tr], lr, epochs, l2)
    oof[te] = sigmoid(Xte @ w + b)

prec, rec, thr = precision_recall_curve(y, oof)
f1 = 2 * prec * rec / (prec + rec + 1e-12)
thr_final = thr[np.argmax(f1)]

# ------------------------------------------------------------
# MODEL PERFORMANCE
# ------------------------------------------------------------
metrics = {
    "Accuracy": accuracy_score(y, oof >= thr_final),
    "Precision": precision_score(y, oof >= thr_final),
    "Recall": recall_score(y, oof >= thr_final),
    "F1": f1_score(y, oof >= thr_final),
    "ROC-AUC": roc_auc_score(y, oof),
}

st.markdown("<div class='section-title'>Model Performance</div>", unsafe_allow_html=True)
cols = st.columns(len(metrics))
for (k,v),c in zip(metrics.items(), cols):
    c.markdown(f"<div class='metric-card'><div class='metric-label'>{k}</div>{v:.3f}</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# MODEL DIAGNOSTICS (✅ AS PER IMAGE)
# ------------------------------------------------------------
st.markdown("<div class='section-title'>Model Diagnostics</div>", unsafe_allow_html=True)
c1,c2,c3 = st.columns(3)

with c1:
    fig,ax = plt.subplots()
    ax.plot(rec, prec)
    ax.set_title("Precision-Recall Curve")
    st.pyplot(fig)

with c2:
    cm = confusion_matrix(y, oof >= thr_final)
    fig,ax = plt.subplots()
    ax.imshow(cm,cmap="Blues")
    for (i,j),v in np.ndenumerate(cm):
        ax.text(j,i,str(v),ha="center",va="center")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

with c3:
    thr_grid = np.linspace(0.05,0.95,50)
    acc_thr = [accuracy_score(y,oof>=t) for t in thr_grid]
    fig,ax = plt.subplots()
    ax.plot(thr_grid,acc_thr)
    ax.axvline(thr_final,color="red",linestyle="--")
    ax.set_title("Accuracy vs Threshold")
    st.pyplot(fig)

# ------------------------------------------------------------
# CHURN RISK DISTRIBUTION
# ------------------------------------------------------------
st.markdown("<div class='section-title'>Churn Risk Distribution</div>", unsafe_allow_html=True)

q_low, q_high = np.percentile(oof,[25,75])
df["Churn_Risk_Level"] = np.where(
    oof>=q_high,"High Churn",
    np.where(oof>=q_low,"Medium Churn","Low Churn")
)

dist = df["Churn_Risk_Level"].value_counts(normalize=True)*100
cols = st.columns(3)
for col,lvl in zip(cols,["High Churn","Medium Churn","Low Churn"]):
    col.markdown(
        f"<div class='metric-card'><div class='metric-label'>{lvl}</div>{dist.get(lvl,0):.1f}%</div>",
        unsafe_allow_html=True
    )

# ------------------------------------------------------------
# ✅ CHURN RISK PREDICTION + CSV DOWNLOAD (RESTORED)
# ------------------------------------------------------------
st.markdown("<div class='section-title'>Churn Risk Prediction</div>", unsafe_allow_html=True)

df_out = df.copy()
df_out["URS"], df_out["BRS"], df_out["SRS"], df_out["CRS"] = URS, BRS, SRS, CRS
df_out["AEGIS_Prob"] = oof.round(4)
df_out["AEGIS_Pred"] = (oof>=thr_final).astype(int)

st.dataframe(df_out.head(20), use_container_width=True)

csv = df_out.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Predictions CSV",
    csv,
    "AEGIS_Divyasri_Churn_Predictions.csv"
)

# ------------------------------------------------------------
# MANUAL CUSTOMER PREDICTION
# ------------------------------------------------------------
st.markdown("<div class='section-title'>Enter Customer Details</div>", unsafe_allow_html=True)

with st.form("manual"):
    tenure = st.number_input("Tenure (Months)",0,72,12)
    monthly = st.number_input("Monthly Charges",0.0,200.0,80.0)
    internet = st.selectbox("Internet Service",["DSL","Fiber optic","No"])
    paperless = st.selectbox("Paperless Billing",["Yes","No"])
    payment = st.selectbox("Payment Method",["Electronic check","Mailed check","Bank transfer","Credit card"])
    contract = st.selectbox("Contract",["Month-to-month","One year","Two year"])
    online = st.selectbox("Online Security",["Yes","No"])
    tech = st.selectbox("Tech Support",["Yes","No"])
    submit = st.form_submit_button("Predict Churn Risk")

if submit:
    URS_i = int(tenure<=12)+int(monthly>=80)+int(internet=="Fiber optic")
    BRS_i = int(paperless=="Yes")+int(payment=="Electronic check")
    SRS_i = int(online=="No")+int(tech=="No")
    CRS_i = {"Month-to-month":2,"One year":1,"Two year":0}[contract]

    scaler = StandardScaler().fit(X_raw)
    w,b = train_model(scaler.transform(X_raw),y,lr,epochs,l2)
    X_u = scaler.transform(np.log1p([[URS_i,BRS_i,SRS_i,CRS_i]]))
    prob = sigmoid(X_u@w+b)[0]

    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>Churn Probability</div>{prob:.3f}</div>",
        unsafe_allow_html=True
    )
