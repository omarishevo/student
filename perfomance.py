import streamlit as st
import pandas as pd

# Page config
st.set_page_config(page_title="Student Performance Analysis", layout="wide")

# Title
st.title("📚 Student Performance Data Explorer")

# Upload the Excel file
uploaded_file = st.file_uploader("Upload your Student Performance Excel file", type=["xlsx"])

if uploaded_file:
    # Load the Excel file into a dataframe (no need for openpyxl here because Streamlit handles it)
    df = pd.read_excel(uploaded_file)

    # Preview the data
    st.subheader("🔍 Preview of Dataset")
    st.dataframe(df.head())

    # Show basic info
    st.subheader("📊 Basic Statistics")
    st.write(df.describe(include='all'))

    # Column filter
    st.subheader("🎯 Column Selector")
    selected_columns = st.multiselect("Select columns to view", df.columns.tolist(), default=df.columns.tolist())
    st.dataframe(df[selected_columns])

    # Missing values
    st.subheader("⚠️ Missing Values Check")
    st.dataframe(df.isnull().sum())

    # Value counts for categorical columns
    st.subheader("🔁 Categorical Value Distributions")
    cat_columns = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_columns:
        st.markdown(f"**{col}**")
        st.dataframe(df[c]()
