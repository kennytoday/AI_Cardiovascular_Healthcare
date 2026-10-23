"""
AI Cardio Health Risk Checker (Improved)
----------------------------------------
Improvements added:
- Personalized recommendations
- Risk vs average comparison
- Input validation warnings
- Lifestyle score
- What-if mode (quit smoking)
- Downloadable result summary
- Cleaner layout
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from framingham import (
    framingham_10y_risk,
    risk_interpretation,
    get_average_risk,
    calculate_lifestyle_score,
)

st.set_page_config(
    page_title="AI Cardio Health Risk Checker",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = ROOT / "models" / "heart_disease_model.pkl"
FEATURES_PATH = ROOT / "models" / "feature_names.json"


@st.cache_resource
def load_ai_model():
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH) as f:
        feature_names = json.load(f)
    return model, feature_names


ai_model, feature_names = load_ai_model()

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


def get_input_warnings(age, sbp, chol, hdl, smoker, diabetes):
    warnings = []
    if sbp >= 180:
        warnings.append("⚠️ Very high blood pressure — seek medical advice soon.")
    elif sbp >= 140:
        warnings.append("⚠️ Blood pressure is elevated.")
    if chol >= 300:
        warnings.append("⚠️ Very high cholesterol — discuss with a doctor.")
    elif chol >= 240:
        warnings.append("⚠️ High cholesterol.")
    if hdl < 35:
        warnings.append("⚠️ Very low HDL (good cholesterol).")
    if smoker:
        warnings.append("⚠️ Smoking significantly increases heart risk.")
    if diabetes:
        warnings.append("⚠️ Diabetes is an important risk factor.")
    if age >= 70:
        warnings.append("ℹ️ Age is a strong risk factor — regular check-ups are important.")
    return warnings


def basic_recommendations(risk_pct, smoker, diabetes, sbp, chol, hdl, lifestyle_score):
    recs = []
    if risk_pct >= 20:
        recs.append("🔴 **High 10-year risk.** Schedule a visit with your doctor soon to discuss prevention strategies.")
    elif risk_pct >= 10:
        recs.append("🟡 **Intermediate risk.** Focus on lifestyle changes and talk to your clinician.")
    else:
        recs.append("🟢 **Lower risk.** Keep up the healthy habits and continue regular check-ups.")

    if smoker:
        recs.append("• **Quit smoking** — this is the single most powerful change you can make.")
    if diabetes:
        recs.append("• Optimize blood sugar control and maintain a healthy weight.")
    if sbp >= 140:
        recs.append("• Blood pressure is elevated. Reduce salt, increase activity, and follow medical advice.")
    if chol >= 240 or hdl < 40:
        recs.append("• Improve cholesterol: more fiber, less saturated fat, regular exercise.")

    if lifestyle_score < 60:
        recs.append("• Your lifestyle score is low — small changes can make a big difference.")
    elif lifestyle_score >= 80:
        recs.append("• Great lifestyle score! Keep doing what you're doing.")

    recs.append("• Aim for 150 minutes of moderate aerobic activity per week.")
    recs.append("• Eat a Mediterranean-style diet (vegetables, fruits, whole grains, healthy fats).")
    recs.append("• Maintain a healthy weight, limit alcohol, manage stress, and get good sleep.")
    recs.append("• This tool is for education only. Always consult a healthcare professional.")
    return recs


def generate_summary_text(result_data: dict) -> str:
    lines = [
        "AI Cardio Health Risk Checker - Result Summary",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"10-Year CVD Risk: {result_data.get('risk_pct', 'N/A')}%",
        f"Category: {result_data.get('category', 'N/A')}",
        f"Average risk for age/sex: ~{result_data.get('avg_risk', 'N/A')}%",
        f"Lifestyle Score: {result_data.get('lifestyle', 'N/A')}/100",
        "",
        "Recommendations:",
    ]
    for rec in result_data.get("recs", []):
        clean = rec.replace("**", "").replace("• ", "- ")
        lines.append(clean)
    lines.append("")
    lines.append("Disclaimer: Educational tool only. Not medical advice.")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────
st.title("❤️ AI Cardio Health Risk Checker")
st.caption("Educational tool • Framingham equations + AI model • Not for clinical use")

st.error(
    """
    ⚠️ **IMPORTANT MEDICAL DISCLAIMER**  
    This is **NOT** a medical device and does **NOT** provide medical advice, diagnosis, or treatment.  
    Results are for **educational purposes only**. Always consult a qualified healthcare professional.
    """
)

mode = st.radio(
    "Choose Mode",
    ["🟢 Basic Mode (Framingham)", "🤖 Advanced Mode (AI Model)", "📊 Both"],
    horizontal=True,
)

st.markdown("---")

if mode.startswith("🟢") or mode.startswith("📊"):
    st.subheader("Basic Mode – Framingham 10-year CVD Risk")

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
        what_if = st.checkbox("What if I quit smoking?", value=False, disabled=not smoker)
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
        avg_risk = get_average_risk(age, sex)
        lifestyle = calculate_lifestyle_score(smoker, diabetes, sbp, total_chol, hdl, bp_treated)

        # What-if calculation
        what_if_risk = None
        if what_if and smoker:
            what_if_result = framingham_10y_risk(
                age=age, sex=sex.lower(), total_chol=total_chol, hdl=hdl,
                sbp=sbp, smoker=False, diabetes=diabetes, bp_treated=bp_treated,
            )
            what_if_risk = what_if_result["risk_percent"]

        warnings = get_input_warnings(age, sbp, total_chol, hdl, smoker, diabetes)
        if warnings:
            for w in warnings:
                st.warning(w)

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
                    <p style="margin:0.5rem 0 0 0;"><strong>Average for your age/sex:</strong> ~{avg_risk}%</p>
                    <p style="margin:0.2rem 0 0 0;"><strong>Lifestyle Score:</strong> {lifestyle}/100</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if what_if_risk is not None:
                st.success(f"**What-if (quit smoking):** Risk drops to **{what_if_risk}%**")
            st.plotly_chart(create_gauge(risk_pct, "10-Year CVD Risk (%)"), use_container_width=True)

        with c2:
            st.subheader("Recommendations")
            recs = basic_recommendations(risk_pct, smoker, diabetes, sbp, total_chol, hdl, lifestyle)
            for rec in recs:
                st.markdown(rec)

            # Download summary
            summary_data = {
                "risk_pct": risk_pct,
                "category": category,
                "avg_risk": avg_risk,
                "lifestyle": lifestyle,
                "recs": recs,
            }
            summary_text = generate_summary_text(summary_data)
            st.download_button(
                label="📥 Download Result Summary",
                data=summary_text,
                file_name="cardio_risk_summary.txt",
                mime="text/plain",
            )

        with st.expander("Technical details (Framingham equation)"):
            st.json(result["details"])

if mode.startswith("🤖") or mode.startswith("📊"):
    if mode.startswith("📊"):
        st.markdown("---")
    st.subheader("Advanced Mode – AI (Random Forest) Prediction")

    with st.sidebar:
        st.header("Clinical Inputs (AI Mode)")
        age_ai = st.slider("Age", 29, 77, 55, key="ai_age")
        sex_ai = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", index=1, key="ai_sex")
        trestbps = st.number_input("Resting BP (mm Hg)", 80, 200, 130, key="ai_bp")
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 240, key="ai_chol")
        fbs = st.selectbox("Fasting BS > 120", [0, 1], format_func=lambda x: "Yes" if x else "No", key="ai_fbs")
        thalach = st.slider("Max Heart Rate", 70, 220, 150, key="ai_hr")
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal", 3: "Asymptomatic"}[x], key="ai_cp")
        restecg = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}[x], key="ai_ecg")
        exang = st.selectbox("Exercise Angina", [0, 1], format_func=lambda x: "Yes" if x else "No", key="ai_exang")
        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.5, 1.0, 0.1, key="ai_oldpeak")
        slope = st.selectbox("ST Slope", [0, 1, 2], format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x], key="ai_slope")
        ca = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3, 4], index=0, key="ai_ca")
        thal = st.selectbox("Thalassemia", [1, 2, 3], format_func=lambda x: {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}[x], index=1, key="ai_thal")
        calculate_ai = st.button("🔍 Calculate AI Risk", type="primary", use_container_width=True)

    if calculate_ai:
        input_dict = {
            "age": age_ai, "sex": sex_ai, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
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
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.plotly_chart(create_ai_gauge(prob), use_container_width=True)

        with c2:
            st.subheader("AI Recommendations")
            if prob >= 0.6:
                st.markdown("🔴 **Higher probability.** Please consult a doctor for proper evaluation.")
            elif prob >= 0.3:
                st.markdown("🟡 **Moderate probability.** Discuss with your doctor and optimize lifestyle.")
            else:
                st.markdown("🟢 **Lower probability.** Continue healthy habits.")
            st.markdown("• This AI model is trained on research data and is **not** a diagnostic tool.")

        st.subheader("Feature Importance")
        imp = pd.DataFrame({"Feature": feature_names, "Importance": ai_model.feature_importances_}).sort_values("Importance")
        fig = go.Figure(go.Bar(x=imp["Importance"], y=imp["Feature"], orientation="h", marker_color="steelblue"))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Educational project • Framingham 2008 + Random Forest on UCI data • Not for clinical use")
