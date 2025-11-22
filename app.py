import streamlit as st
import pandas as pd
import numpy as np
import os, io, joblib, time
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# -----------------------
# CONFIG & STYLES
# -----------------------
st.set_page_config(page_title="Mine Detection ", layout="wide", initial_sidebar_state="expanded")

# Colorful theme CSS
st.markdown(
    """
    <style>
    :root{
      --bg: #0f1724;
      --card: #0b1220;
      --accent1: #29b6f6;
      --accent2: #7c4dff;
      --muted: #9aa7bf;
      --glass: rgba(255,255,255,0.03);
    }
    body {background: linear-gradient(180deg,#071026 0%, #081325 100%); color: #e6eef8;}
    .app-title {font-size:28px; font-weight:700; color:var(--accent1);}
    .app-sub {color:var(--muted); margin-top:-8px; margin-bottom:12px;}
    .card {background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); border:1px solid rgba(255,255,255,0.03); border-radius:12px; padding:14px;}
    .kpi {font-size:20px; font-weight:700; color:var(--accent2);}
    .small {font-size:13px; color:var(--muted);}
    .stButton>button {background: linear-gradient(90deg,var(--accent1), var(--accent2)); color:white; border:none; padding:8px 14px;}
    .download-btn button {background: #ff6b6b; color:white;}
    .metric-card {background: linear-gradient(90deg, rgba(124,77,255,0.12), rgba(41,182,246,0.08)); padding:10px; border-radius:8px;}
    .footer {color:var(--muted); font-size:12px; margin-top:20px;}
    </style>
    """, unsafe_allow_html=True
)

# Header
st.markdown('<div class="app-title">🔎 Mine Detection app</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Feature-rich, colorful and production-like demo for sonar-based rock vs mine classification</div>', unsafe_allow_html=True)

# -----------------------
# Paths and Globals
# -----------------------
DATA_PATH = "sonar_data.csv"
MODEL_DIR = "saved_models_v2"
os.makedirs(MODEL_DIR, exist_ok=True)

AVAILABLE_MODELS = {
    "Logistic Regression": LogisticRegression,
    "K-Nearest Neighbors": KNeighborsClassifier,
    "Support Vector Machine": SVC,
    "Random Forest": RandomForestClassifier
}

# -----------------------
# Utility functions
# -----------------------
@st.cache_data(show_spinner=False)
def load_default_data(path=DATA_PATH):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, header=None)
    return df

def safe_instantiate(name, **params):
    cls = AVAILABLE_MODELS.get(name, LogisticRegression)
    if name == "Support Vector Machine":
        # ensure probability True for predict_proba
        params.setdefault("probability", True)
    try:
        return cls(**params)
    except Exception:
        # fallback to defaults if params invalid
        return cls()

def evaluate(mdl, X_test, y_test):
    y_pred = mdl.predict(X_test)
    probs = mdl.predict_proba(X_test)[:,1] if hasattr(mdl, "predict_proba") else None
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, target_names=["R (rock)","M (mine)"], output_dict=True)
    auc = float(roc_auc_score(y_test, probs)) if probs is not None else None
    return {"acc": acc, "cm": cm, "report": report, "auc": auc, "probs": probs, "y_pred": y_pred}

def plot_confusion(cm, labels=["R","M"]):
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="mako", ax=ax, cbar=False)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    st.pyplot(fig, use_container_width=True)

def plot_roc(y_test, probs):
    if probs is None:
        return
    fpr, tpr, _ = roc_curve(y_test, probs)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color="#6a5acd", label="ROC")
    ax.plot([0,1],[0,1], linestyle="--", color="#7f8c8d")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig, use_container_width=True)

# -----------------------
# Sidebar - Data & Model controls
# -----------------------
st.sidebar.header("Data & Model Control")
uploaded = st.sidebar.file_uploader("Upload sonar CSV (60 features + label)", type=["csv"])
use_default = st.sidebar.checkbox("Use local sonar_data.csv", value=True)
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded, header=None)
        st.sidebar.success("Uploaded dataset")
    except Exception as e:
        st.sidebar.error(f"Upload error: {e}")
        df = None
else:
    df = load_default_data() if use_default else None

if df is None:
    st.sidebar.warning("No dataset loaded. Please upload or place 'sonar_data.csv' in the app folder.")
    st.stop()

st.sidebar.markdown("---")
model_name = st.sidebar.selectbox("Model", list(AVAILABLE_MODELS.keys()))
st.sidebar.markdown("### Hyperparameters")
if model_name == "K-Nearest Neighbors":
    k = st.sidebar.slider("K (neighbors)", 1, 25, 5)
else:
    k = 5
if model_name == "Support Vector Machine":
    svm_kernel = st.sidebar.selectbox("Kernel", ["rbf","linear","poly"])
    svm_c = st.sidebar.slider("C (regularization)", 0.1, 10.0, 1.0)
else:
    svm_kernel, svm_c = "rbf", 1.0
if model_name == "Random Forest":
    rf_estimators = st.sidebar.slider("Estimators", 10, 300, 100, step=10)
else:
    rf_estimators = 100

apply_scaler = st.sidebar.checkbox("Apply Standard Scaler", value=True)
test_frac = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = int(st.sidebar.number_input("Random state", value=42, min_value=0))
st.sidebar.markdown("---")
if st.sidebar.button("Train selected model"):
    st.session_state["train_now"] = True
if st.sidebar.button("Compare all models"):
    st.session_state["compare_all"] = True
if st.sidebar.button("Load saved model"):
    st.session_state["load_saved"] = True

# -----------------------
# Main layout - Top summary cards
# -----------------------
col1, col2, col3 = st.columns([3,2,2])
with col1:
    st.markdown('<div class="card"><div style="display:flex;justify-content:space-between;align-items:center"><div><b>Dataset</b><div class="small">UCI Sonar (or uploaded)</div></div><div class="metric-card"><div class="kpi">'+str(df.shape[0])+' samples</div></div></div></div>', unsafe_allow_html=True)
with col2:
    labels = df.iloc[:, -1].value_counts().to_dict()
    st.markdown('<div class="card"><b>Classes</b><div class="small">Distribution</div></div>', unsafe_allow_html=True)
    st.bar_chart(pd.Series(labels))
with col3:
    st.markdown('<div class="card"><b>Features</b><div class="small">60 numeric features</div></div>', unsafe_allow_html=True)
    st.markdown('<div style="height:6px"></div>')

# -----------------------
# Tabs for features
# -----------------------
tab1, tab2, tab3, tab4 = st.tabs(["Modeling", "Visualize", "Explain", "Predict"])

# -----------------------
# Modeling Tab
# -----------------------
with tab1:
    st.header("Modeling & Evaluation")
    X = df.iloc[:, :60].astype(float).values
    y = np.where(df.iloc[:, 60].values == "M", 1, 0)
    if st.session_state.get("compare_all", False):
        st.info("Training and comparing all models (fast defaults)...")
        models_to_try = {
            "Logistic Regression": {"cls": LogisticRegression, "params": {"max_iter":1000}},
            "KNN": {"cls": KNeighborsClassifier, "params": {"n_neighbors":5}},
            "SVM": {"cls": SVC, "params": {"probability":True, "kernel":"rbf", "C":1.0}},
            "Random Forest": {"cls": RandomForestClassifier, "params": {"n_estimators":100, "random_state":42}}
        }
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state=random_state, stratify=y)
        if apply_scaler:
            scaler_all = StandardScaler(); X_train = scaler_all.fit_transform(X_train); X_test = scaler_all.transform(X_test)
        results = []
        for name, meta in models_to_try.items():
            mdl = meta["cls"](**meta["params"])
            mdl.fit(X_train, y_train)
            ev = evaluate(mdl, X_test, y_test)
            results.append({"Model": name, "Accuracy": ev["acc"], "AUC": ev["auc"] if ev["auc"] is not None else 0.0})
            # save model
            joblib.dump(mdl, os.path.join(MODEL_DIR, f"{name.replace(' ','_')}.pkl"))
        resdf = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)
        st.success("Comparison complete")
        st.dataframe(resdf.style.format({"Accuracy":"{:.3f}", "AUC":"{:.3f}"}))
        st.session_state["compare_all"] = False

    if st.session_state.get("train_now", False):
        # build params for selected model
        params = {}
        if model_name == "K-Nearest Neighbors":
            params["n_neighbors"] = k
        if model_name == "Support Vector Machine":
            params["kernel"] = svm_kernel; params["C"] = svm_c; params["probability"] = True
        if model_name == "Random Forest":
            params["n_estimators"] = rf_estimators; params["random_state"] = 42
        if model_name == "Logistic Regression":
            params["max_iter"] = 2000
        # train selected model
        st.info(f"Training {model_name} ...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state=random_state, stratify=y)
        scaler = None
        if apply_scaler:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        model = safe_instantiate(model_name, **params)
        model.fit(X_train, y_train)
        ev = evaluate(model, X_test, y_test)
        st.success(f"Training finished — Test accuracy: {ev['acc']:.3f}")
        colA, colB = st.columns([2,3])
        with colA:
            st.markdown('<div class="card"><b>Metrics</b></div>', unsafe_allow_html=True)
            st.metric("Accuracy", f"{ev['acc']:.3f}")
            if ev["auc"] is not None:
                st.metric("ROC AUC", f"{ev['auc']:.3f}")
            st.write("Classification report:")
            st.json(ev["report"])
        with colB:
            st.markdown('<div class="card"><b>Confusion Matrix</b></div>', unsafe_allow_html=True)
            plot_confusion(ev["cm"])
            if ev["probs"] is not None:
                st.markdown('<div class="card"><b>ROC Curve</b></div>', unsafe_allow_html=True)
                plot_roc(y_test, ev["probs"])
        # save model and scaler
        save_name = st.text_input("Filename to save model (no extension)", value=model_name.replace(" ","_"))
        if st.button("Save model to disk"):
            joblib.dump(model, os.path.join(MODEL_DIR, save_name + ".pkl"))
            if scaler is not None:
                joblib.dump(scaler, os.path.join(MODEL_DIR, save_name + "_scaler.pkl"))
            st.success(f"Saved {save_name}.pkl in {MODEL_DIR}")
        st.session_state["train_now"] = False

# -----------------------
# Visualize Tab
# -----------------------
with tab2:
    st.header("Visualizations & Dimensionality Reduction")
    st.subheader("Feature distribution and correlation")
    viz_col1, viz_col2 = st.columns([2,3])
    with viz_col1:
        idx = st.selectbox("Feature index to view", list(range(60)), index=0)
        fig, ax = plt.subplots()
        sns.histplot(df.iloc[:, idx].astype(float), ax=ax, kde=True, color="#7c4dff")
        ax.set_title(f"Distribution of feature f{idx}")
        st.pyplot(fig, use_container_width=True)
    with viz_col2:
        if st.checkbox("Show feature correlation heatmap"):
            fig2, ax2 = plt.subplots(figsize=(8,6))
            corr = df.iloc[:,:60].astype(float).corr()
            sns.heatmap(corr, cmap="coolwarm", ax=ax2)
            st.pyplot(fig2, use_container_width=True)
    st.subheader("PCA projection")
    sample_n = st.slider("Number of samples for PCA", 30, min(208, df.shape[0]), 120)
    pca_btn = st.button("Run PCA projection")
    if pca_btn:
        X = df.iloc[:,:60].astype(float).values
        y = np.where(df.iloc[:,60].values=="M",1,0)
        idxs = np.random.choice(range(X.shape[0]), sample_n, replace=False)
        X_sub = X[idxs]
        y_sub = y[idxs]
        pca = PCA(n_components=2)
        proj = pca.fit_transform(X_sub)
        fig3, ax3 = plt.subplots()
        scatter = ax3.scatter(proj[:,0], proj[:,1], c=y_sub, cmap="Spectral", alpha=0.9)
        ax3.set_title("PCA projection (2D)")
        st.pyplot(fig3, use_container_width=True)

# -----------------------
# Explain Tab
# -----------------------
with tab3:
    st.header("Explainability & Feature Importance")
    st.write("Load a saved model to inspect feature coefficients/importances.")
    saved_models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
    model_to_inspect = st.selectbox("Choose saved model", options=["None"] + saved_models)
    if model_to_inspect != "None":
        mdl = joblib.load(os.path.join(MODEL_DIR, model_to_inspect))
        if hasattr(mdl, "coef_"):
            coefs = mdl.coef_[0]
            top_idx = np.argsort(np.abs(coefs))[-12:][::-1]
            df_coefs = pd.DataFrame({"feature": [f"f{i}" for i in top_idx], "coef": coefs[top_idx]})
            st.table(df_coefs.style.format({"coef":"{:.4f}"}))
        elif hasattr(mdl, "feature_importances_"):
            imp = mdl.feature_importances_
            top_idx = np.argsort(imp)[-12:][::-1]
            df_imp = pd.DataFrame({"feature": [f"f{i}" for i in top_idx], "importance": imp[top_idx]})
            st.table(df_imp.style.format({"importance":"{:.4f}"}))
        else:
            st.info("Model does not expose coefficients or feature importances.")

# -----------------------
# Predict Tab
# -----------------------
with tab4:
    st.header("Predict — Single sample or Batch")
    st.write("Load a saved model (or train & save), then input a sample or upload CSV.")
    saved_models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
    if not saved_models:
        st.warning("No saved models found. Train and save a model first.")
    else:
        chosen = st.selectbox("Choose model for prediction", saved_models)
        threshold = st.slider("Decision threshold for 'M' (if probability available)", 0.1, 0.9, 0.5, 0.05)
        mdl = joblib.load(os.path.join(MODEL_DIR, chosen))
        scaler_f = os.path.join(MODEL_DIR, chosen.replace(".pkl", "_scaler.pkl"))
        scaler = None
        if os.path.exists(scaler_f):
            scaler = joblib.load(scaler_f)
        c1, c2 = st.columns(2)
        with c1:
            manual = st.text_area("Paste 60 comma-separated feature values", height=160)
            if st.button("Predict sample now"):
                if not manual.strip():
                    st.warning("Paste values first")
                else:
                    try:
                        vals = [float(x.strip()) for x in manual.split(",") if x.strip()!='']
                        if len(vals) != 60:
                            st.error("Expected 60 values")
                        else:
                            x = np.array(vals).reshape(1,-1)
                            if scaler is not None:
                                x = scaler.transform(x)
                            if hasattr(mdl, "predict_proba"):
                                prob = mdl.predict_proba(x)[0][1]
                                label = "M (mine)" if prob >= threshold else "R (rock)"
                                st.success(f"Prediction: {label} (prob: {prob:.3f})")
                            else:
                                pred = mdl.predict(x)[0]
                                label = "M (mine)" if pred==1 else "R (rock)"
                                st.success(f"Prediction: {label}")
                    except Exception as e:
                        st.error(f"Error: {e}")
        with c2:
            upload_pred = st.file_uploader("Upload CSV (rows = 60 features) for batch prediction", type=["csv"])
            if upload_pred is not None:
                try:
                    df_new = pd.read_csv(upload_pred, header=None)
                    if df_new.shape[1] != 60:
                        st.error("CSV must have exactly 60 columns")
                    else:
                        X_new = df_new.values.astype(float)
                        if scaler is not None:
                            X_new = scaler.transform(X_new)
                        preds = mdl.predict(X_new)
                        probs = mdl.predict_proba(X_new)[:,1] if hasattr(mdl, "predict_proba") else [None]*len(preds)
                        out = df_new.copy()
                        out["Prediction"] = ["M (mine)" if p==1 else "R (rock)" for p in preds]
                        out["Prob"] = [round(float(p),3) if p is not None else "" for p in probs]
                        st.dataframe(out.head())
                        csv_out = out.to_csv(index=False).encode()
                        st.download_button("Download predictions CSV", csv_out, "predictions.csv", "text/csv")
                except Exception as e:
                    st.error(f"Error processing file: {e}")

# Footer
st.markdown('<div class="footer">Built with ❤️ — ADITYA JAIN and team.</div>', unsafe_allow_html=True)
