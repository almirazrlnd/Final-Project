import streamlit as st
import pandas as pd
import plotly.express as px

TARGET = "loan_status"  # 1 = Approved, 0 = Not Approved


# -----------------------------
# Styling: Modern Card UI
# -----------------------------
def inject_css():
    st.markdown(
        """
        <style>
        /* Page spacing */
        .block-container { 
            padding-top: 4.5rem !important; 
            padding-bottom: 2.5rem; 
        }

        /* Sidebar tweaks */
        section[data-testid="stSidebar"] { 
            border-right: 1px solid rgba(0,0,0,0.06); 
        }

        /* Modern cards */
        .card {
            background: #ffffff;
            border: 1px solid rgba(0,0,0,0.06);
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.06);
        }
        .card .label {
            font-size: 0.9rem;
            color: rgba(0,0,0,0.55);
            margin-bottom: 6px;
        }
        .card .value {
            font-size: 1.65rem;
            font-weight: 800;
            line-height: 1.1;
            color: #000000 !important; /* ALWAYS BLACK */
        }
        .card .sub {
            margin-top: 8px;
            font-size: 0.9rem;
            color: rgba(0,0,0,0.6);
        }

        .accent {
            border-left: 6px solid #1b5e20;
        }

        /* Section headers */
        .section-title {
            font-size: 1.1rem;
            font-weight: 800;
            margin: 0.25rem 0 0.75rem 0;
        }

        /* Chips */
        .chip {
            display: inline-block;
            padding: 6px 10px;
            margin: 4px 6px 0 0;
            border-radius: 999px;
            background: rgba(165, 214, 167, 0.25);
            color: #66bb6a;
            font-size: 0.85rem;
            font-weight: 600;
            border: 1px solid rgba(102, 187, 106, 0.6);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def card(label: str, value: str, sub: str = ""):
    html = f"""
    <div class="card accent">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# -----------------------------
# Data
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_excel("Loan Approval.xlsx")

    # ---- Derived features (safe) ----
    if "person_age" in df.columns:
        df["age_group"] = pd.cut(
            df["person_age"],
            bins=[0, 24, 34, 44, 54, 64, 200],
            labels=["<25", "25–34", "35–44", "45–54", "55–64", "65+"],
        )

    if "person_income" in df.columns:
        try:
            df["income_level"] = pd.qcut(
                df["person_income"],
                q=4,
                labels=["Low", "Lower-Mid", "Upper-Mid", "High"],
                duplicates="drop",
            )
        except Exception:
            df["income_level"] = "Unknown"

    if "loan_percent_income" in df.columns:
        df["loan_burden"] = pd.cut(
            df["loan_percent_income"],
            bins=[-1, 0.10, 0.20, 0.35, 10],
            labels=["Low", "Moderate", "High", "Very High"],
        )

    if "credit_score" in df.columns:
        df["credit_score_band"] = pd.cut(
            df["credit_score"],
            bins=[0, 579, 669, 739, 799, 900, 2000],
            labels=["Poor", "Fair", "Good", "Very Good", "Excellent", "Outlier"],
        )

    if "person_home_ownership" in df.columns:
        df["home_owner_flag"] = (
            df["person_home_ownership"].astype(str).str.upper().isin(["OWN", "MORTGAGE"]).astype(int)
        )

    # Heuristic risk flag (NOT causal; quick descriptive signal)
    if set(["credit_score", "loan_percent_income", "previous_loan_defaults_on_file"]).issubset(df.columns):
        prev_yes = df["previous_loan_defaults_on_file"].astype(str).str.lower().isin(["yes", "y", "1", "true"])
        df["high_risk_borrower"] = (
            (df["credit_score"] < 600) | (df["loan_percent_income"] > 0.35) | prev_yes
        ).astype(int)

    return df


# -----------------------------
# Risk Insight Helpers
# -----------------------------
def approval_rate_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    tmp = df.groupby(col)[TARGET].mean().reset_index()
    tmp["approval_rate_%"] = tmp[TARGET] * 100
    tmp["count"] = df.groupby(col)[TARGET].size().values
    tmp = tmp.sort_values("approval_rate_%", ascending=True)
    return tmp


def insight_block(title: str, chips: list[str]):
    st.markdown(f'<div class="section-title"> {title}</div>', unsafe_allow_html=True)
    if not chips:
        st.info("Not enough data to generate reliable insights after applying the filters.")
        return
    st.markdown("".join([f'<span class="chip">{c}</span>' for c in chips]), unsafe_allow_html=True)


def quick_insight(text: str):
    st.markdown(f"- {text}")


# -----------------------------
# Adaptive Visual Insight Helpers (auto-follow filters)
# -----------------------------
def safe_pct(x: float) -> str:
    return f"{x:.1f}%"


def top_category_share(series: pd.Series):
    vc = series.value_counts(dropna=False)
    if vc.empty:
        return None
    top = vc.index[0]
    pct = vc.iloc[0] / vc.sum() * 100
    return top, pct, int(vc.iloc[0]), int(vc.sum())


def best_worst_approval(df: pd.DataFrame, group_col: str, target: str, min_count: int):
    if group_col not in df.columns or target not in df.columns:
        return None
    tbl = df.groupby(group_col)[target].agg(["mean", "count"]).reset_index()
    tbl = tbl[tbl["count"] >= min_count].copy()
    if tbl.empty or tbl.shape[0] < 2:
        return None
    tbl["rate"] = tbl["mean"] * 100
    tbl = tbl.sort_values("rate", ascending=True)
    worst = tbl.iloc[0]
    best = tbl.iloc[-1]
    return worst, best


# -----------------------------
# Main Dashboard
# -----------------------------
def chart():
    inject_css()
    st.markdown("## 🏦 Loan Approval Dashboard")

    df = load_data()

    # ===== Sidebar Filters =====
    st.sidebar.header("Filters")

    # categorical
    gender = st.sidebar.multiselect("Gender", sorted(df["person_gender"].dropna().unique().tolist()))
    education = st.sidebar.multiselect("Education", sorted(df["person_education"].dropna().unique().tolist()))
    home = st.sidebar.multiselect("Home Ownership", sorted(df["person_home_ownership"].dropna().unique().tolist()))
    intent = st.sidebar.multiselect("Loan Intent", sorted(df["loan_intent"].dropna().unique().tolist()))
    prev_default = st.sidebar.multiselect(
        "Previous Defaults On File",
        sorted(df["previous_loan_defaults_on_file"].dropna().unique().tolist()),
    )
    status = st.sidebar.selectbox("Loan Status", ["All", 0, 1])

    # derived categorical
    age_group = st.sidebar.multiselect("Age Group", sorted(df["age_group"].dropna().unique().tolist()))
    income_level = st.sidebar.multiselect("Income Level", sorted(df["income_level"].dropna().unique().tolist()))
    loan_burden = st.sidebar.multiselect("Loan Burden", sorted(df["loan_burden"].dropna().unique().tolist()))
    credit_band = st.sidebar.multiselect("Credit Score Band", sorted(df["credit_score_band"].dropna().unique().tolist()))

    # numeric ranges
    st.sidebar.markdown("---")
    st.sidebar.subheader("Numeric Ranges")

    def slider_rng(col, label):
        s = df[col].dropna()
        if s.empty:
            return None
        vmin, vmax = float(s.min()), float(s.max())
        return st.sidebar.slider(label, min_value=vmin, max_value=vmax, value=(vmin, vmax))

    age_rng = slider_rng("person_age", "Age Range")
    income_rng = slider_rng("person_income", "Income Range")
    loan_rng = slider_rng("loan_amnt", "Loan Amount Range")
    int_rng = slider_rng("loan_int_rate", "Interest Rate Range")
    credit_rng = slider_rng("credit_score", "Credit Score Range")
    lpi_rng = slider_rng("loan_percent_income", "Loan % Income Range")

    # ===== Apply filters =====
    filtered = df.copy()

    if gender:
        filtered = filtered[filtered["person_gender"].isin(gender)]
    if education:
        filtered = filtered[filtered["person_education"].isin(education)]
    if home:
        filtered = filtered[filtered["person_home_ownership"].isin(home)]
    if intent:
        filtered = filtered[filtered["loan_intent"].isin(intent)]
    if prev_default:
        filtered = filtered[filtered["previous_loan_defaults_on_file"].isin(prev_default)]
    if status != "All":
        filtered = filtered[filtered[TARGET] == status]

    if age_group:
        filtered = filtered[filtered["age_group"].isin(age_group)]
    if income_level:
        filtered = filtered[filtered["income_level"].isin(income_level)]
    if loan_burden:
        filtered = filtered[filtered["loan_burden"].isin(loan_burden)]
    if credit_band:
        filtered = filtered[filtered["credit_score_band"].isin(credit_band)]

    if age_rng is not None:
        filtered = filtered[(filtered["person_age"] >= age_rng[0]) & (filtered["person_age"] <= age_rng[1])]
    if income_rng is not None:
        filtered = filtered[(filtered["person_income"] >= income_rng[0]) & (filtered["person_income"] <= income_rng[1])]
    if loan_rng is not None:
        filtered = filtered[(filtered["loan_amnt"] >= loan_rng[0]) & (filtered["loan_amnt"] <= loan_rng[1])]
    if int_rng is not None:
        filtered = filtered[(filtered["loan_int_rate"] >= int_rng[0]) & (filtered["loan_int_rate"] <= int_rng[1])]
    if credit_rng is not None:
        filtered = filtered[(filtered["credit_score"] >= credit_rng[0]) & (filtered["credit_score"] <= credit_rng[1])]
    if lpi_rng is not None:
        filtered = filtered[(filtered["loan_percent_income"] >= lpi_rng[0]) & (filtered["loan_percent_income"] <= lpi_rng[1])]

    # ===== Modern KPI Cards =====
    total = int(filtered.shape[0])
    approved = int(filtered[TARGET].sum()) if total else 0
    not_approved = total - approved
    rate = (approved / total * 100) if total else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Total Applications", f"{total:,}", "Records after filters applied")
    with c2:
        card("Approved", f"{approved:,}", "loan_status = 1")
    with c3:
        card("Approval Rate", f"{rate:.2f}%", "Approved / Total")
    with c4:
        card("Not Approved", f"{not_approved:,}", "loan_status = 0")

    st.divider()

    # ===== Risk Insight Section =====
    st.markdown("## Risk Insight")
    st.caption(
        "Insights are heuristic (descriptive), not causal. "
        "The goal is to highlight the most consistent patterns observed in the data."
    )

    if total < 30:
        st.warning(
            "Too few records after filtering (< 30), so insights may be unstable. "
            "Try relaxing the filters."
        )
    else:
        chips_risk = []
        chips_protect = []

        candidates = [
            "previous_loan_defaults_on_file",
            "credit_score_band",
            "loan_burden",
            "income_level",
            "loan_intent",
            "person_home_ownership",
            "age_group",
        ]
        candidates = [c for c in candidates if c in filtered.columns]

        MIN_COUNT = max(15, int(0.03 * total))

        for col in candidates:
            tbl = approval_rate_table(filtered, col)
            tbl = tbl[tbl["count"] >= MIN_COUNT].copy()
            if tbl.empty:
                continue

            worst = tbl.iloc[0]
            best = tbl.iloc[-1]

            chips_risk.append(f"{col}: {worst[col]} → {worst['approval_rate_%']:.1f}% (n={int(worst['count'])})")
            chips_protect.append(f"{col}: {best[col]} → {best['approval_rate_%']:.1f}% (n={int(best['count'])})")

        insight_block("Lowest Approval Rate Segments", chips_risk[:6])
        insight_block("Highest Approval Rate Segments", chips_protect[:6])

        st.markdown("### Explore Risk by Dimension")
        dim = st.selectbox("Select a dimension to view the approval rate", candidates, index=0)
        tbl = approval_rate_table(filtered, dim).sort_values("approval_rate_%", ascending=True)

        fig = px.bar(
            tbl,
            x=dim,
            y="approval_rate_%",
            hover_data=["count"],
            title=f"Approval Rate by {dim}",
            color="approval_rate_%",
            color_continuous_scale=px.colors.sequential.Greens,
        )
        fig.update_layout(xaxis_title=dim, yaxis_title="Approval Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ===== Insights & Recommendations =====
    st.markdown("## Insights & Recommendations")

    approval_rate = (filtered[TARGET].mean() * 100) if len(filtered) > 0 else 0

    st.markdown("### Key Insights")
    st.markdown(
        f"""
- **Overall Approval Rate:** Approximately **{approval_rate:.1f}%** of applications are approved after filtering.
- This suggests a **selective approval process**, prioritizing risk control over loan volume.
"""
    )

    if "credit_score_band" in filtered.columns and len(filtered) > 0:
        cs_tbl = (filtered.groupby("credit_score_band")[TARGET].mean() * 100).sort_values()
        if len(cs_tbl) >= 2:
            worst_cs = cs_tbl.index[0]
            best_cs = cs_tbl.index[-1]
            st.markdown(
                f"""
- **Credit Score Effect:** The **{best_cs}** band has the highest approval rate, while **{worst_cs}** has the lowest.
- Credit score is a strong descriptive indicator in the approval decision pattern.
"""
            )

    if "loan_burden" in filtered.columns and len(filtered) > 0:
        st.markdown(
            """
- **Loan Burden Impact:** Applicants with higher **loan-to-income burden** tend to show lower approval rates.
- This indicates repayment capacity (affordability) matters beyond income alone.
"""
        )

    if "previous_loan_defaults_on_file" in filtered.columns and len(filtered) > 0:
        prev_tbl = (filtered.groupby("previous_loan_defaults_on_file")[TARGET].mean() * 100)
        if len(prev_tbl) >= 2:
            st.markdown(
                """
- **Previous Defaults:** Applicants with prior loan defaults generally have a lower chance of approval.
- Past repayment behavior remains a critical risk-related signal.
"""
            )

    st.markdown("### Recommendations")
    st.markdown(
        """
- **Strengthen credit-based segmentation:** Maintain credit score as a key screening factor, and consider tiered pricing for mid-score applicants.
- **Control loan burden thresholds:** Apply clear loan-to-income limits rather than relying on income alone.
- **Design risk-adjusted products:** For higher-risk segments, adjust loan amount, tenure, or interest rate instead of only rejecting applications.
- **Use the dashboard for policy review:** Monitor approval rate shifts periodically to detect changes in applicant risk profiles.
"""
    )

    st.divider()

    # ===== Data preview =====
    st.markdown("## Data Preview")
    st.dataframe(filtered.head(20), use_container_width=True)

    st.divider()

    # ===== Charts =====
    st.markdown("## Visual Exploration")

    # --- Pie charts (Categorical Distribution) ---
    st.markdown('<div class="section-title"> Categorical Distribution</div>', unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    with p1:
        fig_gender = px.pie(
            filtered,
            names="person_gender",
            title="Gender Distribution",
            color_discrete_sequence=["#1b5e20", "#a5d6a7"],
        )
        st.plotly_chart(fig_gender, use_container_width=True)

    with p2:
        fig_intent = px.pie(
            filtered,
            names="loan_intent",
            title="Loan Intent Distribution",
            color_discrete_sequence=px.colors.sequential.Greens,
        )
        st.plotly_chart(fig_intent, use_container_width=True)

    #  Adaptive insights (Categorical) - BELOW visuals
    st.markdown("### Insights")
    n = len(filtered)
    min_count = max(10, int(0.05 * n))

    if n < 20:
        st.info("Filtered sample is small, so insights may be unstable. Try relaxing filters.")
    else:
        if "person_gender" in filtered.columns:
            top = top_category_share(filtered["person_gender"])
            if top:
                name, pct, cnt, tot = top
                quick_insight(f"Most applications are **{name}**: **{cnt}/{tot}** ({safe_pct(pct)}).")

            bw = best_worst_approval(filtered, "person_gender", TARGET, min_count)
            if bw:
                worst, best = bw
                quick_insight(
                    f"Approval rate by gender: highest **{best['person_gender']}** "
                    f"({safe_pct(best['rate'])}, n={int(best['count'])}) vs lowest **{worst['person_gender']}** "
                    f"({safe_pct(worst['rate'])}, n={int(worst['count'])})."
                )

        if "loan_intent" in filtered.columns:
            top = top_category_share(filtered["loan_intent"])
            if top:
                name, pct, cnt, tot = top
                quick_insight(f"Most common loan purpose is **{name}**: **{cnt}/{tot}** ({safe_pct(pct)}).")

            bw = best_worst_approval(filtered, "loan_intent", TARGET, min_count)
            if bw:
                worst, best = bw
                quick_insight(
                    f"Approval rate by intent: highest **{best['loan_intent']}** "
                    f"({safe_pct(best['rate'])}, n={int(best['count'])}) vs lowest **{worst['loan_intent']}** "
                    f"({safe_pct(worst['rate'])}, n={int(worst['count'])})."
                )

    st.divider()

    # --- Histograms (Numeric Distribution) -> 2 rows ---
    st.markdown('<div class="section-title"> Numeric Distribution</div>', unsafe_allow_html=True)

    # Row 1
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        fig_age = px.histogram(filtered, x="person_age", nbins=30, title="Age", color_discrete_sequence=["#1b5e20"])
        st.plotly_chart(fig_age, use_container_width=True)
    with r1c2:
        fig_income = px.histogram(filtered, x="person_income", nbins=40, title="Income", color_discrete_sequence=["#1b5e20"])
        st.plotly_chart(fig_income, use_container_width=True)

    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

    # Row 2
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        fig_loan = px.histogram(filtered, x="loan_amnt", nbins=40, title="Loan Amount", color_discrete_sequence=["#1b5e20"])
        st.plotly_chart(fig_loan, use_container_width=True)
    with r2c2:
        fig_ir = px.histogram(filtered, x="loan_int_rate", nbins=40, title="Interest Rate", color_discrete_sequence=["#1b5e20"])
        st.plotly_chart(fig_ir, use_container_width=True)

    # Adaptive insights (Numeric) - BELOW visuals
    st.markdown("### Insights")
    if len(filtered) < 10:
        st.info("Not enough filtered records to summarize numeric distributions reliably.")
    else:
        if "person_age" in filtered.columns:
            quick_insight(
                f"Age median **{filtered['person_age'].median():.0f}**, "
                f"IQR **{filtered['person_age'].quantile(0.25):.0f}–{filtered['person_age'].quantile(0.75):.0f}**."
            )
        if "person_income" in filtered.columns:
            quick_insight(
                f"Income median **{filtered['person_income'].median():,.0f}**, "
                f"IQR **{filtered['person_income'].quantile(0.25):,.0f}–{filtered['person_income'].quantile(0.75):,.0f}**."
            )
        if "loan_amnt" in filtered.columns:
            quick_insight(
                f"Loan amount median **{filtered['loan_amnt'].median():,.0f}**, "
                f"IQR **{filtered['loan_amnt'].quantile(0.25):,.0f}–{filtered['loan_amnt'].quantile(0.75):,.0f}**."
            )
        if "loan_int_rate" in filtered.columns:
            quick_insight(
                f"Interest rate median **{filtered['loan_int_rate'].median():.2f}%**, "
                f"IQR **{filtered['loan_int_rate'].quantile(0.25):.2f}%–{filtered['loan_int_rate'].quantile(0.75):.2f}%**."
            )

    st.divider()

    # --- Feature vs Loan Status -> 2x2 + tooltips ---
    st.markdown('<div class="section-title"> Feature vs Loan Status</div>', unsafe_allow_html=True)

    # Row 1
    f1, f2 = st.columns(2)
    with f1:
        fig_cs = px.box(
            filtered,
            x=TARGET,
            y="credit_score",
            title="Credit Score vs Loan Status",
            color_discrete_sequence=["#1b5e20"],
        )
        st.plotly_chart(fig_cs, use_container_width=True)
        st.caption(
            "ℹ️ **Credit Score** reflects an applicant’s historical creditworthiness. "
            "Higher scores generally indicate lower default risk."
        )

    with f2:
        fig_lpi = px.box(
            filtered,
            x=TARGET,
            y="loan_percent_income",
            title="Loan-to-Income Ratio vs Loan Status",
            color_discrete_sequence=["#1b5e20"],
        )
        st.plotly_chart(fig_lpi, use_container_width=True)
        st.caption(
            "ℹ️ **Loan-to-Income Ratio** measures how much of the applicant’s income is allocated to loan repayment. "
            "Lower ratios usually indicate better affordability."
        )

    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

    # Row 2
    f3, f4 = st.columns(2)
    with f3:
        fig_hist = px.box(
            filtered,
            x=TARGET,
            y="cb_person_cred_hist_length",
            title="Credit History Length vs Loan Status",
            color_discrete_sequence=["#1b5e20"],
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.caption(
            "ℹ️ **Credit History Length** represents the duration of an applicant’s credit activity. "
            "Longer histories typically provide more reliable risk assessment."
        )

    with f4:
        fig_rate = px.box(
            filtered,
            x=TARGET,
            y="loan_int_rate",
            title="Interest Rate vs Loan Status",
            color_discrete_sequence=["#1b5e20"],
        )
        st.plotly_chart(fig_rate, use_container_width=True)
        st.caption(
            "ℹ️ **Interest Rate** reflects loan pricing. Higher rates often correspond to higher perceived borrower risk."
        )

    # Adaptive insights (Feature vs Loan Status) - BELOW visuals
    st.markdown("### Insights")
    if len(filtered) < 30 or filtered[TARGET].nunique() < 2:
        st.info("Not enough filtered data to compare Approved vs Not Approved reliably.")
    else:
        def med_by_status(col):
            if col not in filtered.columns:
                return None
            a = filtered.loc[filtered[TARGET] == 1, col].median()
            r = filtered.loc[filtered[TARGET] == 0, col].median()
            return a, r

        cs = med_by_status("credit_score")
        if cs:
            a, r = cs
            quick_insight(f"Approved applicants tend to have higher credit scores (median **{a:.0f}** vs **{r:.0f}**).")

        lpi = med_by_status("loan_percent_income")
        if lpi:
            a, r = lpi
            quick_insight(f"Approved applicants often have lower loan-to-income ratios (median **{a:.2f}** vs **{r:.2f}**).")

        ch = med_by_status("cb_person_cred_hist_length")
        if ch:
            a, r = ch
            quick_insight(f"Approved applicants tend to have longer credit history (median **{a:.1f}** vs **{r:.1f}**).")

        ir = med_by_status("loan_int_rate")
        if ir:
            a, r = ir
            quick_insight(f"Interest rates differ by outcome (median **{a:.2f}%** vs **{r:.2f}%**).")

    st.divider()

    # --- Scatter plots (Relationships) ---
    st.markdown('<div class="section-title"> Relationship</div>', unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        fig_sc1 = px.scatter(
            filtered,
            x="loan_amnt",
            y="person_income",
            color=TARGET,
            title="Loan Amount vs Income",
            color_continuous_scale=px.colors.sequential.Greens,
        )
        st.plotly_chart(fig_sc1, use_container_width=True)

    with s2:
        fig_sc2 = px.scatter(
            filtered,
            x="loan_int_rate",
            y="loan_percent_income",
            color=TARGET,
            title="Interest Rate vs Loan % Income",
            color_continuous_scale=px.colors.sequential.Greens,
        )
        st.plotly_chart(fig_sc2, use_container_width=True)

    # Adaptive insights (Relationships) - BELOW visuals
    st.markdown("### Insights (Relationships)")
    if len(filtered) < 30:
        st.info("Not enough filtered data to estimate relationships reliably.")
    else:
        if {"loan_amnt", "person_income"}.issubset(filtered.columns):
            corr = filtered[["loan_amnt", "person_income"]].corr().iloc[0, 1]
            quick_insight(f"Loan amount vs income correlation: **{corr:.2f}** (based on filtered data).")

        if {"loan_int_rate", "loan_percent_income"}.issubset(filtered.columns):
            corr2 = filtered[["loan_int_rate", "loan_percent_income"]].corr().iloc[0, 1]
            quick_insight(f"Interest rate vs loan-to-income correlation: **{corr2:.2f}** (based on filtered data).")


if __name__ == "__main__":
    chart()




