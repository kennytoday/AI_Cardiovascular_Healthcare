"""
AI Cardio Health Risk Checker
--------------------------------
Two modes:
1. Basic Mode  – Everyday inputs + official Framingham 10-year CVD risk
2. Advanced Mode – Full clinical features + Random Forest AI model
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make src importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from framingham import framingham_10y_risk, risk_interpretation  # noqa: E402

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Cardio Health Risk Checker",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Load AI model
# ──────────────────────────────────────────────
MODEL_PATH = ROOT / "models" / "heart_disease_model.pkl"
FEATURES_PATH = ROOT / "models" / "feature_names.json"


@st.cache_resource
def load_ai_model():
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH) as f:
        feature_names = json.load(f)
    return model, feature_names


ai_model, feature_names = load_ai_model()

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
FEATURE_HELP = {
    "age": "Age in years",
    "sex": "Biological sex",
    "cp": "Chest pain type: 0=Typical angina, 1=Atypical, 2=Non-anginal, 3=Asymptomatic",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl",
    "restecg": "Resting ECG: 0=Normal, 1=ST-T abnormality, 2=LVH",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression induced by exercise",
    "slope": "Slope of peak exercise ST segment",
    "ca": "Number of major vessels (0-3) colored by fluoroscopy",
    "thal": "Thalassemia: 1=Normal, 2=Fixed defect, 3=Reversible defect",
}


def create_gauge(value: float, title: str, max_val: float = 100) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title, "font": {"size": 16}},
            number={"suffix": "%", "font": {"size": 32}},
            gauge={
                "axis": {"range": [0, max_val], "tickwidth": 1},
                "bar": {"color": "darkblue"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 5], "color": "#d4edda"},
                    {"range": [5, 10], "color": "#cce5ff"},
                    {"range": [10, 20], "color": "#fff3cd"},
                    {"range": [20, max_val], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 3},
                    "thickness": 0.75,
                    "value": 20,
                },
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10))
    return fig


def create_ai_gauge(prob: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "AI Heart Disease Probability (%)", "font": {"size": 16}},
            number={"suffix": "%", "font": {"size": 32}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 30], "color": "#d4edda"},
                    {"range": [30, 60], "color": "#fff3cd"},
                    {"range": [60, 100], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 3},
                    "thickness": 0.75,
                    "value": 60,
                },
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10))
    return fig


def get_ai_risk_level(prob: float):
    if prob < 0.30:
        return "Low", "#28a745", "🟢"
    elif prob < 0.60:
        return "Moderate", "#ffc107", "🟡"
    return "High", "#dc3545", "🔴"


def basic_recommendations(risk_pct: float, smoker: bool, diabetes: bool, sbp: float, chol: float, hdl: float):
    recs = []
    if risk_pct >= 20:
        recs.append("🔴 **High 10-year risk.** Schedule a visit with your doctor soon to discuss prevention strategies (lifestyle ± medication).")
    elif risk_pct >= 10:
        recs.append("🟡 **Intermediate risk.** Focus on lifestyle changes and talk to your clinician about whether further testing or therapy is appropriate.")
    else:
        recs.append("🟢 **Lower risk.** Keep up the healthy habits and continue regular check-ups.")

    if smoker:
        recs.append("• **Quit smoking** – the single most powerful change you can make for heart health.")
    if diabetes:
        recs.append("• Optimize blood sugar control and maintain a healthy weight.")
    if sbp >= 140:
        recs.append("• Blood pressure is elevated. Reduce salt, increase activity, manage stress, and follow medical advice.")
    if chol >= 240 or hdl < 40:
        recs.append("• Improve cholesterol profile: more fiber, less saturated fat, regular exercise. Discuss with your doctor.")

    recs.append("• Aim for 150 minutes of moderate aerobic activity per week.")
    recs.append("• Eat a diet rich in vegetables, fruits, whole grains, and healthy fats (Mediterranean style is excellent).")
    recs.append("• Maintain a healthy weight, limit alcohol, manage stress, and get good sleep.")
    recs.append("• This calculator is for education only. Final decisions belong to you and your healthcare team.")
    return recs


def ai_recommendations(prob: float, inputs: dict):
    recs = []
    if prob >= 0.6:
        recs.append("🔴 **Higher probability indicated by the AI model.** Please consult a cardiologist or primary care physician for proper evaluation.")
    elif prob >= 0.3:
        recs.append("🟡 **Moderate probability.** Discuss results with your doctor and consider lifestyle optimization.")
    else:
        recs.append("🟢 **Lower probability according to the model.** Continue healthy habits and regular check-ups.")

    if inputs.get("trestbps", 0) >= 140:
        recs.append("• Elevated resting blood pressure noted.")
    if inputs.get("chol", 0) >= 240:
        recs.append("• Cholesterol is high – dietary changes and medical review recommended.")
    if inputs.get("exang", 0) == 1:
        recs.append("• Exercise-induced angina reported – seek medical advice before intense exercise.")
    recs.append("• General: quit smoking if applicable, regular exercise, heart-healthy diet, healthy weight, stress management.")
    recs.append("• Remember: this AI model is trained on a research dataset and is **not** a clinical diagnostic tool.")
    return recs


# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────
st.title("❤️ AI Cardio Health Risk Checker")
st.caption("Educational tool combining classic Framingham risk equations + modern machine learning")

# Strong disclaimer
st.error(
    """
    ⚠️ **IMPORTANT MEDICAL DISCLAIMER**  
    This is **NOT** a medical device and does **NOT** provide medical advice, diagnosis, or treatment.  
    Results are for **educational / demonstration purposes only**.  
    Always consult a qualified healthcare professional for any health concerns.  
    Do not make medical decisions based solely on this tool.
    """
)

# Mode selector
mode = st.radio(
    "Choose Mode",
    ["🟢 Basic Mode (Framingham 10-year Risk)", "🤖 Advanced Mode (AI Model)"],
    horizontal=True,
    help="Basic Mode uses everyday numbers most people know. Advanced Mode uses the full AI model that needs more clinical data.",
)

st.markdown("---")

# ══════════════════════════════════════════════
# BASIC MODE
# ══════════════════════════════════════════════
if mode.startswith("🟢"):
    st.subheader("Basic Mode – Framingham 10-year CVD Risk")
    st.markdown(
        "Uses the official **2008 Framingham Heart Study** equations (laboratory version). "
        "These are widely used clinical risk equations based on large population data."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age (years)", 30, 79, 55)
        sex = st.selectbox("Sex", ["Male", "Female"])
        smoker = st.selectbox("Current Smoker", ["No", "Yes"]) == "Yes"

    with col2:
        total_chol = st.number_input("Total Cholesterol (mg/dL)", 100, 400, 200, step=1)
        hdl = st.number_input("HDL Cholesterol (mg/dL)", 20, 100, 50, step=1)
        sbp = st.number_input("Systolic Blood Pressure (mm Hg)", 90, 200, 130, step=1)

    with col3:
        diabetes = st.selectbox("Diabetes", ["No", "Yes"]) == "Yes"
        bp_treated = st.selectbox("On blood pressure medication?", ["No", "Yes"]) == "Yes"
        st.write("")  # spacer
        calculate_basic = st.button("🔍 Calculate 10-year Risk", type="primary", use_container_width=True)

    if calculate_basic:
        result = framingham_10y_risk(
            age=age,
            sex=sex.lower(),
            total_chol=total_chol,
            hdl=hdl,
            sbp=sbp,
            smoker=smoker,
            diabetes=diabetes,
            bp_treated=bp_treated,
        )

        risk_pct = result["risk_percent"]
        category = result["risk_category"]
        color = result["color"]
        emoji = result["emoji"]

        st.markdown("---")
        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown(
                f"""
                <div style="padding: 1.4rem; border-radius: 10px; background-color: {color}22; border-left: 6px solid {color};">
                    <h2 style="margin:0; color:{color};">{emoji} {category} Risk</h2>
                    <p style="font-size:1.5rem; margin:0.4rem 0 0 0;">
                        <strong>{risk_pct}%</strong> estimated 10-year CVD risk
                    </p>
                    <p style="margin:0.3rem 0 0 0; opacity:0.85;">
                        {risk_interpretation(risk_pct)}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.plotly_chart(create_gauge(risk_pct, "10-Year CVD Risk (%)"), use_container_width=True)

        with c2:
            st.subheader("Recommendations")
            for rec in basic_recommendations(risk_pct, smoker, diabetes, sbp, total_chol, hdl):
                st.markdown(rec)

        with st.expander("Technical details (Framingham equation)"):
            st.json(result["details"])
            st.caption("Based on D’Agostino et al., Circulation 2008. Educational implementation only.")

    else:
        st.info("Fill in the values above and click **Calculate 10-year Risk**.")

# ══════════════════════════════════════════════
# ADVANCED MODE (AI)
# ══════════════════════════════════════════════
else:
    st.subheader("Advanced Mode – AI (Random Forest) Prediction")
    st.markdown(
        "Uses a machine-learning model trained on the classic UCI Heart Disease dataset. "
        "Many features require medical tests (ECG, stress test, fluoroscopy)."
    )

    with st.sidebar:
        st.header("Clinical Inputs")
        st.caption("Adjust values then click Calculate")

        age = st.slider("Age", 29, 77, 55, help=FEATURE_HELP["age"])
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", index=1)

        trestbps = st.number_input("Resting BP (mm Hg)", 80, 200, 130)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 240)
        fbs = st.selectbox("Fasting BS > 120", [0, 1], format_func=lambda x: "Yes" if x else "No")
        thalach = st.slider("Max Heart Rate", 70, 220, 150)

        cp = st.selectbox(
            "Chest Pain Type",
            [0, 1, 2, 3],
            format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal", 3: "Asymptomatic"}[x],
        )
        restecg = st.selectbox(
            "Resting ECG",
            [0, 1, 2],
            format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}[x],
        )
        exang = st.selectbox("Exercise Angina", [0, 1], format_func=lambda x: "Yes" if x else "No")
        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.5, 1.0, 0.1)
        slope = st.selectbox(
            "ST Slope",
            [0, 1, 2],
            format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x],
        )
        ca = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3, 4], index=0)
        thal = st.selectbox(
            "Thalassemia",
            [1, 2, 3],
            format_func=lambda x: {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}[x],
            index=1,
        )

        calculate_ai = st.button("🔍 Calculate AI Risk", type="primary", use_container_width=True)

    if calculate_ai:
        input_dict = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }
        X = pd.DataFrame([input_dict])[feature_names]
        prob = float(ai_model.predict_proba(X)[0][1])
        pred = int(ai_model.predict(X)[0])

        label, color, emoji = get_ai_risk_level(prob)

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(
                f"""
                <div style="padding: 1.4rem; border-radius: 10px; background-color: {color}22; border-left: 6px solid {color};">
                    <h2 style="margin:0; color:{color};">{emoji} {label} Risk</h2>
                    <p style="font-size:1.4rem; margin:0.4rem 0 0 0;">
                        AI probability: <strong>{prob*100:.1f}%</strong>
                    </p>
                    <p style="margin:0.2rem 0 0 0; opacity:0.8;">
                        Model class: {"Disease indicators present" if pred == 1 else "No disease indicators"}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.plotly_chart(create_ai_gauge(prob), use_container_width=True)

        with c2:
            st.subheader("Recommendations")
            for rec in ai_recommendations(prob, input_dict):
                st.markdown(rec)

        st.subheader("Global Feature Importance (AI Model)")
        imp = pd.DataFrame(
            {"Feature": feature_names, "Importance": ai_model.feature_importances_}
        ).sort_values("Importance")
        fig = go.Figure(go.Bar(x=imp["Importance"], y=imp["Feature"], orientation="h", marker_color="steelblue"))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Input values used"):
            st.json(input_dict)
    else:
        st.info("👈 Adjust the clinical features in the sidebar and click **Calculate AI Risk**.")
        st.markdown(
            """
            **About the AI model**
            - Algorithm: Random Forest (200 trees)
            - Dataset: UCI Heart Disease – Cleveland (303 patients)
            - Test performance: ~82% accuracy, ~0.92 ROC-AUC
            - Many inputs require specialized medical tests
            """
        )

# Footer
st.markdown("---")
st.caption(
    "Educational project • Framingham equations (2008) + Random Forest on UCI data • "
    "Not for clinical use • Always consult a licensed physician"
)
