# ❤️ AI Cardio Health Risk Checker

A complete educational Python project that combines:

1. **Basic Mode** – Official **2008 Framingham Heart Study** 10-year general CVD risk equations (easy everyday inputs)
2. **Advanced Mode** – Machine Learning (Random Forest) trained on the classic UCI Heart Disease dataset

Built with Streamlit for a clean interactive web interface.

> ⚠️ **Strong Medical Disclaimer**  
> This is an **educational / demonstration tool only**. It is **not** a medical device and must **not** be used for diagnosis, treatment, or clinical decision-making. Always consult a qualified healthcare professional.

---

## Project Structure

```
cardio_risk_ai/
├── app.py                  # Main Streamlit application
├── requirements.txt
├── README.md
├── data/
│   └── heart.csv           # UCI Heart Disease dataset
├── models/
│   ├── heart_disease_model.pkl
│   └── feature_names.json
└── src/
    ├── train_model.py      # Retrain the AI model
    └── framingham.py       # Framingham 10-year risk equations
```

---

## Features

### 🟢 Basic Mode (Recommended for most users)
- Inputs most people already know or can get from a routine blood test:
  - Age, Sex
  - Total Cholesterol & HDL
  - Systolic Blood Pressure
  - Smoking status
  - Diabetes
  - Whether on blood pressure medication
- Calculates **10-year general CVD risk** using the official Framingham 2008 laboratory equations
- Risk categories: Low / Borderline / Intermediate / High
- Personalized lifestyle recommendations

### 🤖 Advanced Mode (AI)
- Full 13 clinical features (some require ECG / stress test / fluoroscopy)
- Random Forest model trained on UCI Cleveland data
- ~82% accuracy / ~0.92 ROC-AUC on held-out test set
- Probability of heart disease indicators + feature importance

---

## Quick Start

```bash
cd cardio_risk_ai
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser (usually http://localhost:8501).

### Optional – Retrain the AI model
```bash
python src/train_model.py
```

---

## Framingham Equations

Implemented from the official 2008 Framingham Heart Study coefficients  
(D’Agostino et al., Circulation 2008;117:743-753).

Uses log-transformed age, total cholesterol, HDL, systolic BP (treated vs untreated), smoking, and diabetes.

---

## Limitations

- Framingham equations were derived from a predominantly White North American population and have known limitations in other ethnic groups.
- The AI model is trained on a small 1980s clinical dataset (303 patients). It is for demonstration only.
- Neither mode replaces professional medical assessment, blood work, ECG, or imaging.
- No patient data is stored or sent anywhere – everything runs locally.

---

## Extending the Project

Ideas:
- Add the 2013 ACC/AHA ASCVD (Pooled Cohort) equations
- Add SHAP explanations for individual AI predictions
- Support metric units (mmol/L)
- Deploy to Streamlit Community Cloud or Hugging Face Spaces

---

## License & Credits

- Framingham equations: public research coefficients from the Framingham Heart Study
- Dataset: UCI Machine Learning Repository – Heart Disease Data Set
- Built with: scikit-learn, Streamlit, Plotly, pandas

Use and modify freely for learning and personal experimentation.
.