import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load Data ---
try:
    df = pd.read_csv('student_habits_performance.csv')
    df = df.drop('student_id', axis=1)
except FileNotFoundError:
    st.error("Error: Make sure 'student_habits_performance.csv' is in the same directory as your script.")
    st.stop()

# --- Preprocessing ---
for col in ['gender', 'part_time_job', 'diet_quality', 'parental_education_level', 'internet_quality', 'extracurricular_participation']:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop('exam_score', axis=1)
y = df['exam_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Model Training ---
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# --- Streamlit App ---
st.markdown("""
    <style>
    .big-font {
        font-size:24px !important;
        font-weight: bold;
    }
    .reportview-container {
        background: linear-gradient(to bottom, #66ccff 0%, #ff99cc 100%);
    }
   .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #66ccff 0%, #ff99cc 100%);
    }
    h1 {
        color: #FF4B4B;
        text-shadow: 2px 2px 4px #000000;
    }
    h2 {
        color: #333333;
    }
    .stSlider>div>div>div>div {
        background-color: #FF4B4B;
    }
    .stRadio>div>label {
        color: #333333;
    }
    </style>
    """, unsafe_allow_html=True)

st.title('Student Performance Prediction')

# --- Sidebar ---
st.sidebar.header('Input Features')

age = st.sidebar.slider('Age', 17, 25, 20)
gender = st.sidebar.radio('Gender', [0, 1], index=0, format_func=lambda x: "Female" if x == 0 else "Male")
study_hours = st.sidebar.slider('Study Hours per Day', 0.0, 8.0, 4.0)
social_media = st.sidebar.slider('Social Media Hours', 0.0, 8.0, 2.0)
netflix = st.sidebar.slider('Netflix Hours', 0.0, 8.0, 1.0)
part_time = st.sidebar.radio('Part-time Job', [0, 1], index=0, format_func=lambda x: "No" if x == 0 else "Yes")
attendance = st.sidebar.slider('Attendance Percentage', 0, 100, 80)
sleep = st.sidebar.slider('Sleep Hours', 0.0, 10.0, 7.0)
diet = st.sidebar.radio('Diet Quality', [0, 1, 2], index=1, format_func=lambda x: "Fair" if x == 0 else ("Good" if x == 1 else "Poor"))
exercise = st.sidebar.slider('Exercise Frequency', 0, 7, 3)
parental_education = st.sidebar.radio('Parental Education Level', [0, 1, 2, 3], index=1, format_func=lambda x: "High School" if x == 0 else ("None" if x == 1 else ("Bachelor" if x == 2 else "Master")))
internet = st.sidebar.radio('Internet Quality', [0, 1, 2], index=1, format_func=lambda x: "Average" if x == 0 else ("Good" if x == 1 else "Poor"))
mental_health = st.sidebar.slider('Mental Health Rating', 1, 10, 5)
extra_curricular = st.sidebar.radio('Extracurricular Participation', [0, 1], index=0, format_func=lambda x: "No" if x == 0 else "Yes")

# --- Prediction ---
input_data = pd.DataFrame({
    'age': [age],
    'gender': [gender],
    'study_hours_per_day': [study_hours],
    'social_media_hours': [social_media],
    'netflix_hours': [netflix],
    'part_time_job': [part_time],
    'attendance_percentage': [attendance],
    'sleep_hours': [sleep],
    'diet_quality': [diet],
    'exercise_frequency': [exercise],
    'parental_education_level': [parental_education],
    'internet_quality': [internet],
    'mental_health_rating': [mental_health],
    'extracurricular_participation': [extra_curricular]
})

input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]

# --- Display Prediction ---
st.subheader('Prediction')
st.markdown(f'<p class="big-font">Predicted Exam Score: {prediction:.2f}</p>', unsafe_allow_html=True)

# --- Visualizations ---
st.subheader('Visualizations')

# Correlation Heatmap
st.write("Correlation Heatmap:")
fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='viridis', ax=ax_corr)
st.pyplot(fig_corr)

# Feature Importance
st.write("Feature Importance:")
feat_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
fig_importance, ax_importance = plt.subplots(figsize=(10, 6))
feat_importance.plot(kind='bar', ax=ax_importance, color="#FF4B4B")
ax_importance.set_title('Feature Importance')
ax_importance.set_ylabel('Importance')
st.pyplot(fig_importance)

# Study Hours vs Exam Score
st.write("Study Hours vs Exam Score:")
fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x='study_hours_per_day', y='exam_score', ax=ax_scatter, color="#2E9AFE")
ax_scatter.set_title('Study Hours vs Exam Score')
st.pyplot(fig_scatter)
