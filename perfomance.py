import streamlit as st
import pandas as pd

# Page config
st.set_page_config(page_title="Student Performance Analysis", layout="wide")

# Title
st.title("📚 Student Performance Data Explorer")

# Upload the CSV file
uploaded_file = st.file_uploader("Upload your Student Performance CSV file", type=["csv"])

if uploaded_file:
    try:
        # Read CSV instead of Excel
        df = pd.read_csv(uploaded_file)

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
            st.dataframe(df[col].value_counts())

    except Exception as e:
        st.error(f"❌ An error occurred while processing the file: {e}")
else:
    st.warning("📤 Please upload a student performance CSV file to begin.")

# Footer
st.markdown("---")
st.caption("📌 Built with ❤️ using Streamlit | Basic Analysis Only – No ML or Charts")
