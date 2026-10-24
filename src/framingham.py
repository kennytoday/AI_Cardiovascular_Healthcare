"""
2008 Framingham Heart Study General CVD 10-year Risk equations
(with laboratory values) + helper functions for the improved app.
"""

import math
from typing import Dict


def framingham_10y_risk(
    age: float,
    sex: str,
    total_chol: float,
    hdl: float,
    sbp: float,
    smoker: bool,
    diabetes: bool,
    bp_treated: bool,
) -> Dict:
    """Calculate 10-year general CVD risk using Framingham 2008 equations."""
    sex = sex.lower().strip()
    if sex not in ("male", "female"):
        raise ValueError("sex must be 'male' or 'female'")

    age = max(30, min(79, age))
    total_chol = max(100, min(400, total_chol))
    hdl = max(20, min(100, hdl))
    sbp = max(90, min(200, sbp))

    ln_age = math.log(age)
    ln_chol = math.log(total_chol)
    ln_hdl = math.log(hdl)
    ln_sbp = math.log(sbp)

    if sex == "male":
        beta_age = 3.06117
        beta_chol = 1.12370
        beta_hdl = -0.93263
        beta_sbp_untreated = 1.93303
        beta_sbp_treated = 1.99881
        beta_smoker = 0.65451
        beta_diabetes = 0.57367
        mean_sum = 23.9802
        baseline_survival = 0.88936
    else:
        beta_age = 2.32888
        beta_chol = 1.20904
        beta_hdl = -0.70833
        beta_sbp_untreated = 2.76157
        beta_sbp_treated = 2.82263
        beta_smoker = 0.52873
        beta_diabetes = 0.69154
        mean_sum = 26.1931
        baseline_survival = 0.95012

    sum_bx = (
        beta_age * ln_age
        + beta_chol * ln_chol
        + beta_hdl * ln_hdl
        + (beta_sbp_treated if bp_treated else beta_sbp_untreated) * ln_sbp
        + (beta_smoker if smoker else 0.0)
        + (beta_diabetes if diabetes else 0.0)
    )

    risk = 1.0 - (baseline_survival ** math.exp(sum_bx - mean_sum))
    risk_percent = max(0.0, min(100.0, risk * 100.0))

    if risk_percent < 5:
        category = "Low"
        color = "#28a745"
        emoji = "🟢"
    elif risk_percent < 10:
        category = "Borderline"
        color = "#17a2b8"
        emoji = "🔵"
    elif risk_percent < 20:
        category = "Intermediate"
        color = "#ffc107"
        emoji = "🟡"
    else:
        category = "High"
        color = "#dc3545"
        emoji = "🔴"

    return {
        "risk_percent": round(risk_percent, 1),
        "risk_category": category,
        "color": color,
        "emoji": emoji,
        "details": {
            "sum_bx": round(sum_bx, 4),
            "mean_sum": mean_sum,
            "baseline_survival": baseline_survival,
        },
    }


def risk_interpretation(risk_percent: float) -> str:
    if risk_percent < 5:
        return "Low 10-year risk of general cardiovascular disease."
    if risk_percent < 10:
        return "Borderline risk. Lifestyle optimization is still valuable."
    if risk_percent < 20:
        return "Intermediate risk. Discuss further evaluation and possible preventive therapy with your doctor."
    return "High risk. Prompt discussion with a healthcare professional is recommended."


def get_average_risk(age: float, sex: str) -> float:
    """Rough average 10-year CVD risk by age and sex (educational estimate)."""
    sex = sex.lower()
    if sex == "male":
        if age < 40:
            return 2.0
        elif age < 50:
            return 5.0
        elif age < 60:
            return 10.0
        elif age < 70:
            return 16.0
        else:
            return 22.0
    else:
        if age < 40:
            return 1.0
        elif age < 50:
            return 3.0
        elif age < 60:
            return 7.0
        elif age < 70:
            return 12.0
        else:
            return 18.0


def calculate_lifestyle_score(smoker: bool, diabetes: bool, sbp: float, chol: float, hdl: float, bp_treated: bool) -> int:
    """Simple 0-100 lifestyle/risk factor score (higher = better)."""
    score = 100
    if smoker:
        score -= 25
    if diabetes:
        score -= 15
    if sbp >= 140:
        score -= 15
    elif sbp >= 130:
        score -= 8
    if chol >= 240:
        score -= 12
    elif chol >= 200:
        score -= 6
    if hdl < 40:
        score -= 10
    elif hdl < 50:
        score -= 5
    if bp_treated and sbp >= 140:
        score -= 5
    return max(0, min(100, score))
