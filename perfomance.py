import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 12px; color: white;
        text-align: center; margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa; border-left: 4px solid #667eea;
        padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
    }
    .section-header {
        color: #764ba2; border-bottom: 2px solid #667eea;
        padding-bottom: 0.3rem; margin: 1.5rem 0 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; border: none; padding: 0.6rem 2rem;
        border-radius: 8px; font-weight: 600; width: 100%;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border: 2px solid #667eea; border-radius: 12px;
        padding: 1.5rem; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Load & cache data ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("StudentsPerformance.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("/", "_")
    df["average_score"] = df[["math_score", "reading_score", "writing_score"]].mean(axis=1).round(2)
    df["pass_fail"] = (df["average_score"] >= 60).map({True: "Pass", False: "Fail"})
    return df

@st.cache_data
def prepare_ml(df, target):
    le = LabelEncoder()
    cat_cols = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
    df_enc = df.copy()
    for col in cat_cols:
        df_enc[col] = le.fit_transform(df_enc[col])
    features = cat_cols
    X = df_enc[features]
    y = df_enc[target]
    return train_test_split(X, y, test_size=0.2, random_state=42), features

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=80)
    st.title("🎓 Controls")
    st.markdown("---")

    uploaded = st.file_uploader("📂 Upload your CSV", type=["csv"])
    st.markdown("---")

    st.subheader("🔧 Model Settings")
    target_var = st.selectbox("🎯 Target Score", ["math_score", "reading_score", "writing_score", "average_score"])
    model_choice = st.selectbox("🤖 Algorithm", ["Random Forest", "Gradient Boosting", "Linear Regression"])
    test_size_pct = st.slider("Test Split %", 10, 40, 20)
    n_estimators = st.slider("Trees (RF / GB)", 50, 300, 100, step=50)
    st.markdown("---")
    st.caption("v1.0 · Student Performance ML App")

# ── Load data ──────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv(uploaded) if uploaded else load_data()
    if uploaded:
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("/", "_")
        df["average_score"] = df[["math_score", "reading_score", "writing_score"]].mean(axis=1).round(2)
        df["pass_fail"] = (df["average_score"] >= 60).map({True: "Pass", False: "Fail"})
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎓 Student Performance Predictor</h1>
    <p>Explore data insights and predict student scores using machine learning</p>
</div>
""", unsafe_allow_html=True)

# ── Top KPIs ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("👥 Students", len(df))
k2.metric("📐 Avg Math", f"{df['math_score'].mean():.1f}")
k3.metric("📖 Avg Reading", f"{df['reading_score'].mean():.1f}")
k4.metric("✍️ Avg Writing", f"{df['writing_score'].mean():.1f}")
k5.metric("🏆 Pass Rate", f"{(df['pass_fail']=='Pass').mean()*100:.1f}%")

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Explorer", "📈 Visualisations", "🤖 ML Model", "🔮 Predict"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<h3 class="section-header">Dataset Overview</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(df.head(20), use_container_width=True)
    with col2:
        st.markdown("**Shape**")
        st.info(f"{df.shape[0]} rows × {df.shape[1]} columns")
        st.markdown("**Score Statistics**")
        st.dataframe(df[["math_score", "reading_score", "writing_score", "average_score"]].describe().round(2))

    st.markdown('<h3 class="section-header">Filter & Explore</h3>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    genders = f1.multiselect("Gender", df["gender"].unique(), default=list(df["gender"].unique()))
    groups  = f2.multiselect("Race/Ethnicity", df["race_ethnicity"].unique(), default=list(df["race_ethnicity"].unique()))
    preps   = f3.multiselect("Test Prep", df["test_preparation_course"].unique(), default=list(df["test_preparation_course"].unique()))

    filtered = df[df["gender"].isin(genders) & df["race_ethnicity"].isin(groups) & df["test_preparation_course"].isin(preps)]
    st.success(f"Showing **{len(filtered)}** students after filters")
    st.dataframe(filtered, use_container_width=True, height=300)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<h3 class="section-header">Score Distributions</h3>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ["#667eea", "#764ba2", "#f093fb"]
    for ax, col, color in zip(axes, ["math_score", "reading_score", "writing_score"], colors):
        ax.hist(df[col], bins=20, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(df[col].mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {df[col].mean():.1f}")
        ax.set_title(col.replace("_", " ").title(), fontsize=13, fontweight="bold")
        ax.set_xlabel("Score"); ax.set_ylabel("Count"); ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<h4 class="section-header">Scores by Gender</h4>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        gender_means = df.groupby("gender")[["math_score", "reading_score", "writing_score"]].mean()
        gender_means.plot(kind="bar", ax=ax2, color=["#667eea", "#764ba2", "#f093fb"], edgecolor="white")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0, fontsize=11)
        ax2.set_ylabel("Average Score"); ax2.set_title("Average Scores by Gender")
        ax2.legend(["Math", "Reading", "Writing"])
        st.pyplot(fig2)

    with col_b:
        st.markdown('<h4 class="section-header">Test Prep Impact</h4>', unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        prep_means = df.groupby("test_preparation_course")[["math_score", "reading_score", "writing_score"]].mean()
        prep_means.plot(kind="bar", ax=ax3, color=["#667eea", "#764ba2", "#f093fb"], edgecolor="white")
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0, fontsize=10)
        ax3.set_ylabel("Average Score"); ax3.set_title("Scores by Test Preparation")
        ax3.legend(["Math", "Reading", "Writing"])
        st.pyplot(fig3)

    st.markdown('<h4 class="section-header">Correlation Heatmap</h4>', unsafe_allow_html=True)
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    corr = df[["math_score", "reading_score", "writing_score", "average_score"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdPu", ax=ax4, linewidths=0.5)
    ax4.set_title("Score Correlations")
    st.pyplot(fig4)

    st.markdown('<h4 class="section-header">Average Score by Parental Education</h4>', unsafe_allow_html=True)
    edu_order = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
    edu_means = df.groupby("parental_level_of_education")["average_score"].mean().reindex(edu_order)
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    bars = ax5.bar(edu_means.index, edu_means.values, color="#667eea", alpha=0.85, edgecolor="white")
    ax5.set_xlabel("Education Level"); ax5.set_ylabel("Avg Score"); ax5.set_title("Average Score by Parental Education")
    plt.xticks(rotation=30, ha="right")
    for bar in bars:
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    st.pyplot(fig5)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – ML MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f'<h3 class="section-header">Training: {model_choice} → {target_var}</h3>', unsafe_allow_html=True)

    (X_train, X_test, y_train, y_test), features = prepare_ml(df, target_var)

    # Override split with sidebar slider
    X_train, X_test, y_train, y_test = train_test_split(
        pd.concat([X_train, X_test]), pd.concat([y_train, y_test]),
        test_size=test_size_pct/100, random_state=42
    )

    if model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    elif model_choice == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
    else:
        model = LinearRegression()

    with st.spinner("Training model..."):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    m1, m2, m3 = st.columns(3)
    m1.metric("📉 RMSE", f"{rmse:.2f}")
    m2.metric("📏 MAE",  f"{mae:.2f}")
    m3.metric("📊 R² Score", f"{r2:.4f}")

    col_p, col_f = st.columns(2)

    with col_p:
        st.markdown("**Actual vs Predicted**")
        fig6, ax6 = plt.subplots(figsize=(6, 5))
        ax6.scatter(y_test, y_pred, alpha=0.5, color="#667eea", edgecolors="white", s=40)
        mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax6.plot([mn, mx], [mn, mx], "r--", linewidth=1.5)
        ax6.set_xlabel("Actual"); ax6.set_ylabel("Predicted")
        ax6.set_title(f"Actual vs Predicted — {target_var}")
        st.pyplot(fig6)

    with col_f:
        if model_choice != "Linear Regression":
            st.markdown("**Feature Importances**")
            imp = pd.Series(model.feature_importances_, index=features).sort_values()
            fig7, ax7 = plt.subplots(figsize=(6, 5))
            imp.plot(kind="barh", ax=ax7, color="#764ba2", edgecolor="white")
            ax7.set_title("Feature Importances"); ax7.set_xlabel("Importance")
            st.pyplot(fig7)
        else:
            st.markdown("**Residuals Distribution**")
            residuals = y_test - y_pred
            fig7, ax7 = plt.subplots(figsize=(6, 5))
            ax7.hist(residuals, bins=20, color="#764ba2", alpha=0.8, edgecolor="white")
            ax7.axvline(0, color="red", linestyle="--")
            ax7.set_xlabel("Residual"); ax7.set_ylabel("Count"); ax7.set_title("Residuals")
            st.pyplot(fig7)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<h3 class="section-header">🔮 Predict a Student\'s Score</h3>', unsafe_allow_html=True)
    st.info("Fill in the student details below, then click **Predict**.")

    c1, c2 = st.columns(2)
    with c1:
        p_gender  = st.selectbox("Gender", df["gender"].unique())
        p_group   = st.selectbox("Race/Ethnicity", sorted(df["race_ethnicity"].unique()))
        p_edu     = st.selectbox("Parental Education", ["some high school", "high school", "some college",
                                                         "associate's degree", "bachelor's degree", "master's degree"])
    with c2:
        p_lunch   = st.selectbox("Lunch Type", df["lunch"].unique())
        p_prep    = st.selectbox("Test Preparation", df["test_preparation_course"].unique())
        p_target  = st.selectbox("Score to Predict", ["math_score", "reading_score", "writing_score", "average_score"])

    if st.button("🎯 Predict Score"):
        # Encode using same label encoding approach
        cat_cols = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
        df_enc2 = df.copy()
        le2 = LabelEncoder()
        encoders = {}
        for col in cat_cols:
            df_enc2[col] = le2.fit_transform(df_enc2[col])
            encoders[col] = dict(zip(df[col], df_enc2[col]))

        input_vals = [
            encoders["gender"].get(p_gender, 0),
            encoders["race_ethnicity"].get(p_group, 0),
            encoders["parental_level_of_education"].get(p_edu, 0),
            encoders["lunch"].get(p_lunch, 0),
            encoders["test_preparation_course"].get(p_prep, 0),
        ]
        input_df = pd.DataFrame([input_vals], columns=cat_cols)

        # Retrain on full data for the selected target
        X_all = df_enc2[cat_cols]
        y_all = df_enc2[p_target]
        if model_choice == "Random Forest":
            pred_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        elif model_choice == "Gradient Boosting":
            pred_model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
        else:
            pred_model = LinearRegression()
        pred_model.fit(X_all, y_all)
        prediction = pred_model.predict(input_df)[0]
        prediction = max(0, min(100, prediction))
        grade = "A" if prediction >= 90 else "B" if prediction >= 80 else "C" if prediction >= 70 else "D" if prediction >= 60 else "F"

        st.markdown(f"""
        <div class="prediction-box">
            <h2>Predicted {p_target.replace("_", " ").title()}</h2>
            <h1 style="color:#667eea; font-size:3rem">{prediction:.1f} / 100</h1>
            <h3>Grade: <span style="color:#764ba2">{grade}</span></h3>
            <p>{"✅ Pass" if prediction >= 60 else "❌ Needs Improvement"}</p>
        </div>
        """, unsafe_allow_html=True)

        # Show percentile
        actual_scores = df[p_target]
        percentile = (actual_scores < prediction).mean() * 100
        st.markdown(f"📊 This score is better than **{percentile:.1f}%** of students in the dataset.")
