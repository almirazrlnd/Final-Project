import streamlit as st
import pandas as pd
import plotly.express as px
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from imblearn.over_sampling import SMOTE
import joblib


def ml_model():
    # =========================================================
    # 0) Page config + GREEN THEME (CSS only, logic unchanged)
    # =========================================================
    st.set_page_config(page_title="data/Loan Approval Analysis & Prediction", layout="wide", page_icon="🏦")

    st.markdown(
        """
        <style>
        /* Page spacing (prevents header overlap) */
        .block-container { padding-top: 4.5rem !important; padding-bottom: 2.5rem; }

        /* Sidebar tweaks */
        section[data-testid="stSidebar"] { border-right: 1px solid rgba(0,0,0,0.06); }

        /* Title spacing + green headers */
        h1, h2, h3 { scroll-margin-top: 6rem; color: #1b5e20; }

        /* ===== Clean card styles for text blocks (tidy UI) ===== */
        .soft-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 16px 16px;
            margin-top: 10px;
        }
        .soft-title {
            font-size: 1.15rem;
            font-weight: 900;
            margin: 0 0 10px 0;
            color: #ffffff;
            letter-spacing: 0.2px;
        }
        .soft-text {
            font-size: 0.98rem;
            line-height: 1.6;
            color: rgba(255,255,255,0.85);
            margin: 0;
        }
        .soft-list li {
            margin-bottom: 6px;
            color: rgba(255,255,255,0.85);
            line-height: 1.55;
            font-size: 0.98rem;
        }

        /* ======================================
           METRICS: VALUES WHITE IN ANY MODE
           ====================================== */
        div[data-testid="stMetricLabel"] {
            color: rgba(0,0,0,0.75) !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetricValue"] {
            color: #ffffff !important;     /* ALWAYS WHITE */
            font-weight: 900 !important;
        }
        div[data-testid="stMetricDelta"] {
            color: #a5d6a7 !important;     /* soft green */
            font-weight: 800 !important;
        }
        div[data-testid="stMetric"] {
            background: rgba(27, 94, 32, 0.92) !important; /* deep green */
            border-radius: 14px !important;
            padding: 14px 16px !important;
            border: 1px solid rgba(165, 214, 167, 0.35) !important;
        }
        div[data-testid="metric-container"] {
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
        }
        div[data-testid="stMetric"] {
            width: 100% !important;
        }
        /* Reduce markdown vertical gaps */
        .stMarkdown { margin-bottom: 0.25rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🌲 Random Forest (Loan Approval)")
    st.write("Target: **loan_status** (1 = Approved, 0 = Not Approved)")

    # =========================================================
    # 0) Load dataset
    # =========================================================
    df = pd.read_excel("Loan Approval.xlsx")

    st.write("### Initial Dataset")
    st.write(f"Total rows: **{df.shape[0]}** | Columns: **{df.shape[1]}**")
    st.dataframe(df.head(), use_container_width=True)

    # =========================================================
    # 1) Separate numeric and categorical columns (BEFORE outlier)
    # =========================================================
    numbers = df.select_dtypes(include=["number"]).drop(columns=["loan_status"], errors="ignore").columns
    categories = df.select_dtypes(exclude=["number"]).columns

    # =========================================================
    # 2) Outlier handling (IQR) - use numeric columns excluding loan_status
    # =========================================================
    st.write("### 1. Outlier Detection (IQR Method)")

    if len(numbers) == 0:
        st.warning("No numeric columns available for outlier detection.")
    else:
        Q1 = df[numbers].quantile(0.25)
        Q3 = df[numbers].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        rows_before = df.shape[0]

        df = df[~((df[numbers] < lower_bound) | (df[numbers] > upper_bound)).any(axis=1)]

        rows_after = df.shape[0]

        st.write("**Before Outlier Removal**")
        b1, b2 = st.columns(2)

        with b1:
            st.metric("Total Rows", f"{rows_before:,}")

        with b2:
            st.metric("Rows Removed", f"{rows_before - rows_after:,}")

        st.write("**After Outlier Removal**")
        c1, c2 = st.columns(2)

        with c1:
            st.metric("Remaining Rows", f"{rows_after:,}")

        with c2:
            st.metric(
                "Retention Rate",
                f"{(rows_after / rows_before * 100):.2f}%"
            )
    st.markdown("#### Interpretation")
    st.markdown(
        """
Outlier removal reduces extreme values that may distort model learning.
A high **retention rate** indicates that most data is preserved, achieving a good balance between **data quality** and **data quantity**.
"""
)
    st.dataframe(df.head(), use_container_width=True)

    # =========================================================
    # 3) Correlation Heatmap (AFTER outlier removal)
    # =========================================================
    st.write("### 2. Linear Correlation Between Numeric Columns")
    num_corr = df.select_dtypes(include=["number"]).drop(columns=["loan_status"], errors="ignore")

    if num_corr.shape[1] >= 2:
        corr = num_corr.corr().round(2)
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Correlation Heatmap",
            # ✅ GREEN THEME HEATMAP
            color_continuous_scale=px.colors.sequential.Greens,
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=60, b=10),
            height=420,
            title_font=dict(size=18),
            coloraxis_colorbar=dict(len=0.7),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ✅ Description-style insight (added, no logic changes)
        st.markdown("#### Interpretation")
        st.markdown(
            """
This heatmap shows the **linear relationships** among numeric features after outlier handling.
- Higher values mean stronger **positive** correlation.
- Values near 0 indicate weak linear relationship.

Correlation can suggest **redundant information**, but correlation does not imply causation.
"""
        )
    else:
        st.warning("Not enough numeric columns to build a heatmap.")

    # =========================================================
    # 4) Drop person_age (if exists) + recompute numbers/categories
    # =========================================================
    if "person_age" in df.columns:
        df = df.drop(columns=["person_age"])

    numbers = df.select_dtypes(include=["number"]).drop(columns=["loan_status"], errors="ignore").columns
    categories = df.select_dtypes(exclude=["number"]).columns

    # =========================================================
    # 5) Copy dataset (AFTER THIS, do MinMax scaling)
    # =========================================================
    df_select = df.copy()

    # =========================================================
    # 6) MinMax Scaling (BEFORE Encoding)
    #    Numeric only, exclude target & engineered flags
    # =========================================================
    st.write("### 3. Normalization (MinMaxScaler)")

    scaler = MinMaxScaler()

    numbers_for_scaling = df_select.drop(
        columns=[
            "loan_status",
            "stable_employment",
            "has_previous_default",
            "high_risk_borrower",
            "home_owner_flag",
        ],
        errors="ignore",
    ).select_dtypes(include=["number"]).columns.tolist()

    # keep before scaling (for visualization)
    before_scale = df_select[numbers_for_scaling].copy() if len(numbers_for_scaling) else pd.DataFrame()

    # apply scaling
    if len(numbers_for_scaling) > 0:
        df_select.loc[:, numbers_for_scaling] = scaler.fit_transform(df_select[numbers_for_scaling])

    # after scaling
    after_scale = df_select[numbers_for_scaling].copy() if len(numbers_for_scaling) else pd.DataFrame()

    # =========================================================
    # 6A) Visualization of scaling
    # =========================================================
    st.write("### Normalization Visualization (MinMaxScaler)")

    if len(numbers_for_scaling) == 0:
        st.warning("No numeric columns to visualize.")
    else:
        mode = st.radio("Visualization Mode", ["Select Columns", "All Columns"], horizontal=True)

        if mode == "Select Columns":
            selected_cols = st.multiselect(
                "Select numeric columns",
                numbers_for_scaling,
                default=numbers_for_scaling[:2] if len(numbers_for_scaling) >= 2 else numbers_for_scaling,
            )
            for col in selected_cols:
                st.write(f"#### Distribution: **{col}**")
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.histogram(
                        before_scale, x=col, nbins=30, title=f"Before Scaling ({col})",
                        color_discrete_sequence=["#1b5e20"]
                    )
                    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=320)
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig = px.histogram(
                        after_scale, x=col, nbins=30, title=f"After Scaling ({col})",
                        color_discrete_sequence=["#1b5e20"]
                    )
                    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=320)
                    st.plotly_chart(fig, use_container_width=True)

                # ✅ Description-style insight
                st.markdown("##### Interpretation")
                st.markdown(
                    f"""
MinMax scaling transforms **{col}** into a value range from **0 to 1**.
This helps the model treat numeric features consistently even if original units differ.
"""
                )
        else:
            n_cols = 3
            rows = math.ceil(len(numbers_for_scaling) / n_cols)
            for i in range(rows):
                cols = st.columns(n_cols)
                for j in range(n_cols):
                    idx = i * n_cols + j
                    if idx < len(numbers_for_scaling):
                        col_name = numbers_for_scaling[idx]
                        fig = px.histogram(
                            after_scale, x=col_name, nbins=30, title=f"{col_name} (After)",
                            color_discrete_sequence=["#1b5e20"]
                        )
                        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=280)
                        cols[j].plotly_chart(fig, use_container_width=True)

            # ✅ Description-style insight
            st.markdown("#### Interpretation")
            st.markdown(
                """
All numeric columns are now displayed on a **consistent 0–1 scale**.
This reduces dominance of large-range features during training.
"""
            )

    # =========================================================
    # 7) One-Hot Encoding (AFTER scaling)
    # =========================================================
    st.write("### 4. Feature Encoding (One-Hot)")
    df_encoded = pd.get_dummies(df_select, drop_first=True)

    st.write("**After Encoding**")
    c1, c2 = st.columns(2)

    with c1:
        st.metric("Total Features (X)", f"{df_encoded.shape[1] - 1:,}")

    with c2:
        st.metric("Target Column (y)", "loan_status")


    # ✅ Description-style insight
    st.markdown("#### Interpretation")
    st.markdown(
        """
One-hot encoding converts categorical variables into numeric features so they can be used by the model.
Dropping the first category avoids redundant information and reduces feature multicollinearity.
After encoding, the dataset is ready for modeling.
"""
    )

    # =========================================================
    # 8) X & y + check imbalance
    # =========================================================
    X = df_encoded.drop("loan_status", axis=1)
    y = df_encoded["loan_status"]

    st.write("### Loan Status Distribution (Before Split)")

    counts = y.value_counts()
    total = len(y)
    approval_rate = (y.mean() * 100) if total else 0.0

    st.write("**Class Distribution**")
    c1, c2 = st.columns(2)

    with c1:
        st.metric("Label 0 (Not Approved)", f"{int(counts.get(0, 0)):,}")

    with c2:
        st.metric("Label 1 (Approved)", f"{int(counts.get(1, 0)):,}")

    st.write("**Base Rate Summary**")
    d1, d2 = st.columns(2)

    with d1:
        st.metric("Approval Base Rate", f"{approval_rate:.2f}%")

    with d2:
        st.metric("Class Imbalance Ratio", f"{counts.get(0,1)/max(counts.get(1,1),1):.1f} : 1")

    # Interpretation (clean, consistent)
    st.markdown("#### Interpretation")
    st.markdown(
        f"""
The approval base rate (**{approval_rate:.2f}%**) indicates a clear class imbalance.
Without balancing techniques, models may favor the majority class (Not Approved),
leading to poor detection of eligible applicants.
"""
    )

    # =========================================================
    # 9) Train–Test Split
    # =========================================================
    st.write("### 5. Train–Test Split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    st.write("**Train–Test Split Summary**")
    c1, c2 = st.columns(2)

    with c1:
        st.metric("X_train (Training Features)", f"{len(X_train):,}")   

    with c2:
        st.metric("X_test (Testing Features)", f"{len(X_test):,}")

    # (Optional) keep your interpretation text below, unchanged
    st.markdown("#### Interpretation")
    st.markdown(
        """
The dataset is split into training and testing sets.
The test set simulates unseen data to evaluate real-world model performance.
Stratified splitting preserves the original class distribution in both sets.
"""
)

    # =========================================================
    # 10) SMOTE (train only)
    # =========================================================
    st.write("### 6. Handling Imbalanced Classes (SMOTE)")
    b1, b2 = st.columns(2)
    with b1:
        st.write("**Before SMOTE**")
        st.metric("Label 0 (Not Approved)", int((y_train == 0).sum()))
        st.metric("Label 1 (Approved)", int((y_train == 1).sum()))

    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

    with b2:
        st.write("**After SMOTE**")
        st.metric("Label 0 (Not Approved)", int((y_train_bal == 0).sum()))
        st.metric("Label 1 (Approved)", int((y_train_bal == 1).sum()))

    # ✅ Description-style insight
    st.markdown("#### Interpretation")
    st.markdown(
        """
SMOTE increases the minority class (Approved=1) **only in the training data** to reduce imbalance.
This often improves recall, but may increase false positives if the model becomes too permissive.
"""
    )

    # =========================================================
    # 11) Train Random Forest
    # =========================================================
    st.write("### 7. Modeling (Random Forest)")
    rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(X_train_bal, y_train_bal)

    train_acc = rf.score(X_train_bal, y_train_bal)

    st.write("**Training Performance**")
    c1, c2 = st.columns(2)

    with c1:
        st.metric("Training Accuracy", f"{train_acc*100:.2f}%")

    with c2:
        st.metric("Model Type", "Random Forest")

    st.markdown("#### Interpretation")
    st.markdown(
        f"""
The model achieves **{train_acc*100:.2f}% training accuracy**, indicating it fits the training data very well.
This is expected for Random Forest models, but evaluation on the **test set** is essential
to ensure the model generalizes and is not overfitting.
"""
)


    # Feature importance
    st.write("### 8. Feature Importance (Top 5)")
    imp = pd.DataFrame(
        {"Feature": X_train.columns, "Importance": rf.feature_importances_}
    ).sort_values("Importance", ascending=False).head()

    fig = px.bar(
        imp, x="Feature", y="Importance", title="Top 5 Feature Importance",
        color="Importance", color_continuous_scale=px.colors.sequential.Greens
    )
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420, title_font=dict(size=18))
    st.plotly_chart(fig, use_container_width=True)

    # ✅ Description-style insight (auto)
    st.markdown("#### Interpretation")
    if len(imp) >= 3:
        t = imp["Feature"].tolist()
        st.markdown(
            f"""
The model relies most heavily on **{t[0]}**, followed by **{t[1]}** and **{t[2]}**.
These features contribute most to how the Random Forest separates approved vs not approved cases.

> Note: Importance reflects model behavior, not causality.
"""
        )
    elif len(imp) > 0:
        st.markdown(
            f"The most important feature shown is **{imp.iloc[0]['Feature']}** (based on model splits)."
        )
    else:
        st.markdown("No feature importance could be computed.")

    # =========================================================
    # 12) Model Evaluation (test) + Confusion Matrix
    # =========================================================
    st.write("### 9. Model Evaluation (Test Set)")
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]  # Prob Approved (1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    # ✅ safer ROC AUC (keeps flow; prevents crash if a class disappears)
    roc_auc = roc_auc_score(y_test, y_prob) if y_test.nunique() > 1 else float("nan")

    # ---- tidy top row: confusion matrix + metrics ----
    col1, col2 = st.columns([6, 4], gap="large")

    with col1:
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Not Approved (0)", "Actual Approved (1)"],
            columns=["Pred Not Approved (0)", "Pred Approved (1)"],
        )
        fig = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale=px.colors.sequential.Greens,
            labels=dict(x="Predicted", y="Actual"),
            aspect="auto",
            title="Confusion Matrix – Random Forest",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=650, title_font=dict(size=18))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div style="display:flex; flex-direction:column; gap:18px;">', unsafe_allow_html=True)

        st.metric("Accuracy", f"{accuracy*100:.2f}%")
        st.metric("Precision (Approved=1)", f"{precision*100:.2f}%")
        st.metric("Recall (Approved=1)", f"{recall*100:.2f}%")
        st.metric("F1 Score", f"{f1*100:.2f}%")
        st.metric("ROC AUC", f"{roc_auc*100:.2f}%")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---- tidy second row: Interpretation + Performance summary ----
    st.markdown("")  # small spacer
    a, b = st.columns(2, gap="large")

    with a:
        tn, fp, fn, tp = cm.ravel()
        st.markdown(
            f"""
            <div class="soft-card">
                <div class="soft-title"> Interpretation</div>
                <ul class="soft-list">
                    <li><b>{tp}</b> correctly approved (<b>True Positives</b>)</li>
                    <li><b>{tn}</b> correctly rejected (<b>True Negatives</b>)</li>
                    <li><b>{fp}</b> approved but should be rejected (<b>False Positives</b>)</li>
                    <li><b>{fn}</b> rejected but should be approved (<b>False Negatives</b>)</li>
                </ul>
                <p class="soft-text">
                    This means the model currently produces more <b>{"false rejections" if fn > fp else "false approvals"}</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with b:
        st.markdown(
            f"""
            <div class="soft-card">
                <div class="soft-title"> Performance Summary</div>
                <ul class="soft-list">
                    <li><b>Accuracy ({accuracy*100:.2f}%)</b>: overall correctness.</li>
                    <li><b>Precision ({precision*100:.2f}%)</b>: reliability of predicted approvals (risk control).</li>
                    <li><b>Recall ({recall*100:.2f}%)</b>: ability to capture true approvals (approval coverage).</li>
                    <li><b>F1 ({f1*100:.2f}%)</b>: balance between precision and recall.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ✅ Overall recommendation (placed after visuals, flow unchanged)
    st.markdown("### Overall Insight & Recommendation")
    emphasis = "risk control (**Precision**)"
    st.markdown(
        f"""
Based on current results, the model emphasizes **{emphasis}**.

**Recommendations (Risk Control):**  
To reduce **false approvals (False Positives)**, the model should prioritize **Precision**.

- **Higher Precision** means when the model says “Approved”, it is more likely to be correct.  
- This helps protect the lender from approving risky borrowers and reduces potential default exposure.  
- Precision can be improved by using a **higher approval threshold** (stricter approval decision), or adding a **manual review band** for borderline cases.  
- Because higher Precision can lower Recall (more false rejections), you can balance it by:
  **manual review**, stronger documentation requirements, or tiered loan terms instead of auto-rejecting.

In short: **Precision-first = safer approvals and better risk control.**
"""
)


    # =========================================================
    # 13) Save trained model files for prediction app
    # (wording changed only; outputs unchanged)
    # =========================================================
    joblib.dump(rf, "model_rf.pkl", compress=3)
    joblib.dump(list(X_train.columns), "model_features_rf.pkl")
    joblib.dump(numbers_for_scaling, "numeric_columns_rf.pkl")  # scaled numeric cols before encoding
    joblib.dump(scaler, "scaler_rf.pkl")


if __name__ == "__main__":
    ml_model()


