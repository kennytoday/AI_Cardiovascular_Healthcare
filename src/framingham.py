"""
2008 Framingham Heart Study General CVD 10-year Risk equations
(with laboratory values).

Source: D’Agostino et al., Circulation 2008;117:743-753
Official coefficients from framinghamheartstudy.org
"""

import math
from typing import Dict, Tuple


def framingham_10y_risk(
    age: float,
    sex: str,  # "male" or "female"
    total_chol: float,  # mg/dL
    hdl: float,  # mg/dL
    sbp: float,  # mmHg
    smoker: bool,
    diabetes: bool,
    bp_treated: bool,
) -> Dict:
    """
    Calculate 10-year general CVD risk using Framingham 2008 equations.

    Returns a dict with:
        risk_percent : float (0-100)
        risk_category: str
        details      : intermediate values
    """
    sex = sex.lower().strip()
    if sex not in ("male", "female"):
        raise ValueError("sex must be 'male' or 'female'")

    # Clamp reasonable ranges to avoid math domain errors
    age = max(30, min(79, age))
    total_chol = max(100, min(400, total_chol))
    hdl = max(20, min(100, hdl))
    sbp = max(90, min(200, sbp))

    ln_age = math.log(age)
    ln_chol = math.log(total_chol)
    ln_hdl = math.log(hdl)
    ln_sbp = math.log(sbp)

    if sex == "male":
        # Men coefficients
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
        # Women coefficients
        beta_age = 2.32888
        beta_chol = 1.20904
        beta_hdl = -0.70833
        beta_sbp_untreated = 2.76157
        beta_sbp_treated = 2.82263
        beta_smoker = 0.52873
        beta_diabetes = 0.69154
        mean_sum = 26.1931
        baseline_survival = 0.95012

    # Linear predictor
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

    # Categories commonly used
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
