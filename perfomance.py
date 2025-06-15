import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="🎓 Student Performance Dashboard", layout="wide")

st.title("🎓 Student Performance Analysis & Prediction App")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your student performance Excel dataset", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    st.subheader("📊 Raw Dataset Preview")
    st.dataframe(df.head())

    # Cleaning step (basic)
    df_cleaned = df.dropna()  # remove rows with missing values

    # Encoding categorical variables
    label_encoders = {}
    for column in df_cleaned.select_dtypes(include=['object']):
        le = LabelEncoder()
        df_cleaned[column] = le.fit_transform(df_cleaned[column])
        label_encoders[column] = le

    st.subheader("🧹 Cleaned & Encoded Dataset")
    st.dataframe(df_cleaned.head())

    # Target column selection
    target_col = st.selectbox("🎯 Select the Target Column to Predict", options=df_cleaned.columns)

    features = df_cleaned.drop(columns=[target_col])
    target = df_cleaned[target_col]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Model training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("📈 Model Performance")
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Feature importance
    st.subheader("🔥 Feature Importance")
    importances = pd.Series(model.feature_importances_, index=features.columns).sort_values(ascending=False)
    st.bar_chart(importances)

    # Prediction form
    st.subheader("🧠 Try Making a Prediction")

    input_data = {}
    for col in features.columns:
        val = st.number_input(f"{col}", value=float(df_cleaned[col].mean()))
        input_data[col] = val

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    decoded_target = prediction
    if target_col in label_encoders:
        decoded_target = label_encoders[target_col].inverse_transform([prediction])[0]

    st.success(f"✅ Predicted {target_col}: {decoded_target}")

else:
    st.info("📎 Please upload a valid Excel file (.xlsx) to get started.")
