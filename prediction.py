import streamlit as st
import pandas as pd
import joblib

# ============================================================
# Loan Approval Prediction App (Random Forest, precision-oriented)
# Target:
#   loan_status = 1  -> APPROVED
#   loan_status = 0  -> NOT APPROVED
#
# Precision Mode = stricter approvals -> fewer False Positives
# + 3-level decision band: Approve / Manual Review / Reject
# ============================================================

THRESHOLD_DEFAULT = 0.65  # stricter -> higher precision, fewer false positives


def inject_css():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 4.5rem !important; padding-bottom: 2.5rem; }
        section[data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,0.08); }
        h1, h2, h3 { scroll-margin-top: 6rem; }

        .soft-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 16px;
            padding: 16px 16px;
            margin-top: 10px;
        }
        .soft-title {
            font-size: 1.10rem;
            font-weight: 900;
            margin: 0 0 10px 0;
            color: #ffffff;
            letter-spacing: 0.2px;
        }
        .soft-text {
            font-size: 0.98rem;
            line-height: 1.6;
            color: rgba(255,255,255,0.88);
            margin: 0;
        }

        .pill {
            display: inline-block;
            padding: 8px 12px;
            border-radius: 999px;
            font-weight: 900;
            font-size: 0.95rem;
            border: 1px solid rgba(165, 214, 167, 0.40);
            background: rgba(27, 94, 32, 0.75);
            color: #ffffff;
            margin-right: 8px;
        }
        .pill-amber {
            border: 1px solid rgba(255, 224, 178, 0.45);
            background: rgba(245, 124, 0, 0.50);
            color: #ffffff;
        }
        .pill-red {
            border: 1px solid rgba(255, 205, 210, 0.45);
            background: rgba(198, 40, 40, 0.55);
            color: #ffffff;
        }

        .risk-badge {
            display: inline-block;
            padding: 7px 11px;
            border-radius: 999px;
            font-weight: 900;
            font-size: 0.90rem;
            border: 1px solid rgba(255,255,255,0.18);
            margin-right: 8px;
            letter-spacing: 0.2px;
        }
        .risk-low  { background: rgba(46, 125, 50, 0.35); color: #c8e6c9; border-color: rgba(200,230,201,0.35); }
        .risk-med  { background: rgba(245, 124, 0, 0.28); color: #ffe0b2; border-color: rgba(255,224,178,0.35); }
        .risk-high { background: rgba(198, 40, 40, 0.28); color: #ffcdd2; border-color: rgba(255,205,210,0.35); }

        div[data-testid="stMetricLabel"] {
            color: rgba(255,255,255,0.80) !important;
            font-weight: 750 !important;
        }
        div[data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-weight: 900 !important;
        }
        div[data-testid="stMetricDelta"] {
            color: #a5d6a7 !important;
            font-weight: 800 !important;
        }
        div[data-testid="stMetric"] {
            background: rgba(27, 94, 32, 0.92) !important;
            border-radius: 14px !important;
            padding: 14px 16px !important;
            border: 1px solid rgba(165, 214, 167, 0.35) !important;
        }
        div[data-testid="metric-container"] {
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
        }

        .stMarkdown { margin-bottom: 0.25rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def soft_card(title: str, body_html: str):
    st.markdown(
        f"""
        <div class="soft-card">
            <div class="soft-title">{title}</div>
            <div class="soft-text">{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_label(prob_approval: float):
    if prob_approval >= 0.75:
        return "Low Risk", "risk-low", "Strong approval signal based on model patterns."
    if prob_approval >= 0.50:
        return "Medium Risk", "risk-med", "Borderline case — consider extra review or documentation."
    return "High Risk", "risk-high", "Low approval signal — higher chance of rejection based on model patterns."


def decision_band(prob_approval: float, approve_threshold: float, review_band: float):
    """
    3-level band:
      - Approve if prob >= approve_threshold
      - Manual Review if in [approve_threshold - review_band, approve_threshold)
      - Reject if below (approve_threshold - review_band)
    """
    review_threshold = max(0.0, approve_threshold - review_band)
    if prob_approval >= approve_threshold:
        return "APPROVE", review_threshold
    if prob_approval >= review_threshold:
        return "MANUAL_REVIEW", review_threshold
    return "REJECT", review_threshold


def prediction_app():
    st.set_page_config(page_title="Loan Approval Analysis & Prediction", layout="wide", page_icon="🏦")
    inject_css()

    st.title("🏦 Loan Approval Prediction")
    st.write(
        "This app predicts the **probability of a loan being APPROVED** using a trained **Random Forest** model.\n\n"
        "**Precision Mode** helps reduce false approvals (False Positives) by requiring a higher score to approve.\n"
        "We also add a 3-level decision band: **Approve / Manual Review / Reject**."
    )

    # ======================
    # Load trained model files
    # ======================
    st.sidebar.header("Model Files")
    try:
        model = joblib.load("model_rf.pkl")
        feature_names = joblib.load("model_features_rf.pkl")
        numeric_cols = joblib.load("numeric_columns_rf.pkl")
        scaler = joblib.load("scaler_rf.pkl")
        st.sidebar.success("Model files loaded ✅")
    except Exception as e:
        st.sidebar.error("Failed to load model files ❌")
        st.error(
            "❌ Cannot load model/metadata files.\n\n"
            "Make sure these files exist in the SAME folder as the app:\n"
            "- model_rf.pkl\n- model_features_rf.pkl\n- numeric_columns_rf.pkl\n- scaler_rf.pkl"
        )
        st.exception(e)
        return

    st.sidebar.write("---")
    st.sidebar.subheader("Decision Settings")

    approve_threshold = st.sidebar.slider(
        "Approve Threshold (higher = stricter approvals)",
        min_value=0.30,
        max_value=0.90,
        value=float(THRESHOLD_DEFAULT),
        step=0.01,
    )

    review_band = st.sidebar.slider(
        "Manual Review Band Width",
        min_value=0.05,
        max_value=0.30,
        value=0.15,
        step=0.01,
        help="Applicants slightly below the approval threshold go to Manual Review instead of Reject.",
    )

    st.sidebar.write("---")
    st.sidebar.subheader("Optional Business Rule")
    rule_reject_payment_problem = st.sidebar.toggle(
        "Always reject if borrower had past payment problems",
        value=True,
        help="This does NOT change the model. It enforces a policy on top of the model decision.",
    )

    # ======================
    # Input options
    # ======================
    gender_opts = ["Male", "Female"]
    edu_opts = ["High School", "Associate", "Bachelor", "Master", "PhD"]
    home_opts = ["RENT", "OWN", "MORTGAGE", "OTHER"]
    intent_opts = ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
    payment_problem_opts = ["Yes", "No"]

    # ======================
    # Input form
    # ======================
    st.subheader("Borrower Information")
    st.write("Enter borrower details to estimate the **approval probability**.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        person_gender = st.selectbox("Gender", gender_opts)
    with c2:
        person_income = st.number_input("Annual Income", min_value=0.0, value=60000.0, step=1000.0)
    with c3:
        person_emp_exp = st.number_input("Work Experience (years)", min_value=0, max_value=50, value=5)
    with c4:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=680)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        person_education = st.selectbox("Education", edu_opts)
    with c2:
        person_home_ownership = st.selectbox("Home Ownership", home_opts)
    with c3:
        loan_intent = st.selectbox("Loan Purpose", intent_opts)
    with c4:
        previous_payment_problems = st.selectbox("Past Payment Problems", payment_problem_opts)

    c1, c2, c3 = st.columns(3)
    with c1:
        loan_amnt = st.number_input("Loan Amount", min_value=0.0, value=10000.0, step=500.0)
    with c2:
        loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=40.0, value=12.0, step=0.1)
    with c3:
        cb_person_cred_hist_length = st.number_input(
            "Credit History Length (years)",
            min_value=0.0, max_value=50.0, value=5.0
        )

    # ======================
    # Derived feature: loan_percent_income
    # ======================
    loan_percent_income = (loan_amnt / person_income) if person_income > 0 else 0.0
    soft_card(
        "Auto-calculated Feature",
        f"""
        <span class="pill">loan_percent_income</span>
        <b>{loan_percent_income:.4f}</b><br/>
        This is the loan burden relative to income (loan amount ÷ annual income).
        """,
    )

    # ======================
    # Build 1-row input dataframe
    # NOTE: you dropped age in the model -> DO NOT include person_age here.
    # ======================
    user_df = pd.DataFrame({
        "person_gender": [person_gender],
        "person_income": [person_income],
        "person_emp_exp": [person_emp_exp],
        "person_emp_experience": [person_emp_exp],  # safe compatibility
        "person_education": [person_education],
        "person_home_ownership": [person_home_ownership],
        "loan_intent": [loan_intent],
        "previous_loan_defaults_on_file": [previous_payment_problems],
        "loan_amnt": [loan_amnt],
        "loan_int_rate": [loan_int_rate],
        "cb_person_cred_hist_length": [cb_person_cred_hist_length],
        "credit_score": [credit_score],
        "loan_percent_income": [loan_percent_income],
    })

    # ======================
    # Preprocess (same approach as training)
    # ======================
    user_processed = pd.get_dummies(user_df, drop_first=True)

    for col in feature_names:
        if col not in user_processed.columns:
            user_processed[col] = 0

    user_processed = user_processed[feature_names]

    cols_to_scale = [c for c in numeric_cols if c in user_processed.columns]
    if cols_to_scale:
        user_processed[cols_to_scale] = scaler.transform(user_processed[cols_to_scale])

    # ======================
    # Predict
    # ======================
    st.write("")
    if st.button("Predict Loan Decision"):
        prob_approval = float(model.predict_proba(user_processed)[0][1])  # P(Approved=1)
        prob_percent = prob_approval * 100

        band, review_threshold = decision_band(prob_approval, approve_threshold, review_band)

        rule_triggered = (previous_payment_problems.strip().lower() == "yes")
        if rule_reject_payment_problem and rule_triggered:
            band = "REJECT"

        # Risk label (based on probability)
        r_name, r_class, r_desc = risk_label(prob_approval)

        st.markdown("## Result")

        left, mid, right = st.columns([3, 2, 5], gap="large")

        # --- Decision card (3-level) ---
        with left:
            if band == "APPROVE":
                pill = '<span class="pill">✅ APPROVE</span>'
                desc = "High confidence approval (precision-oriented)."
            elif band == "MANUAL_REVIEW":
                pill = '<span class="pill pill-amber">🟧 MANUAL REVIEW</span>'
                desc = "Close to the threshold — send for manual checks instead of auto-reject."
            else:
                pill = '<span class="pill pill-red">❌ REJECT</span>'
                desc = "Below review threshold or blocked by policy."

            policy_note = ""
            if rule_reject_payment_problem and rule_triggered:
                policy_note = "<br/><span style='color:rgba(255,255,255,0.78)'>Policy applied: past payment problems → Reject</span>"

            soft_card(
                "Decision",
                f"""
                {pill}<br/>
                <span style="color:rgba(255,255,255,0.82);font-size:0.95rem;">
                    Approve threshold: <b>{approve_threshold:.2f}</b><br/>
                    Review threshold: <b>{review_threshold:.2f}</b>
                </span>
                {policy_note}
                <br/><br/>
                <span style="color:rgba(255,255,255,0.86)">{desc}</span>
                """,
            )

        # --- Probability (card, not metric) ---
        with mid:
            prob_model = prob_approval * 100
            soft_card(
                "Model Approval Probability",
                f"""
                <div style="font-size:2.4rem;font-weight:900;color:#ffffff;line-height:1.1;">
                    {prob_model:.2f}%
                </div>
                <div style="margin-top:8px;color:rgba(255,255,255,0.80);font-size:0.95rem;">
                    This is the model’s score <b>before</b> applying rules/policy.
                </div>
                """,
            )

        # --- Risk label card ---
        with right:
            r_name, r_class, r_desc = risk_label(prob_approval)
            soft_card(
                "Risk Label",
                f"""
                <span class="risk-badge {r_class}">{r_name}</span><br/>
                <span style="color:rgba(255,255,255,0.86)">{r_desc}</span><br/><br/>
                <span style="color:rgba(255,255,255,0.78)">
                    Precision mode reduces false approvals by requiring a higher score to approve.
                    Manual Review avoids unnecessary auto-rejects near the cutoff.
                </span>
                """,
            )

        # --- Clear message when policy overrides model ---
        if rule_reject_payment_problem and rule_triggered:
            soft_card(
                "Why Reject Even With High Probability?",
                f"""
                The model predicted <b>{prob_model:.2f}%</b> approval probability, but the final decision is <b>REJECT</b>
                because the policy “past payment problems → reject” is enabled.
                <br/><br/>
                If you want the decision to follow the model score, turn off the policy in the sidebar.
                """,
            )


        with st.expander("View Raw Input"):
            st.dataframe(user_df, use_container_width=True)

        with st.expander("View Processed Data (Encoding & Scaling)"):
            st.dataframe(user_processed, use_container_width=True)


if __name__ == "__main__":
    prediction_app()



















