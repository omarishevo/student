import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import altair as alt

# --- Helper Functions ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def initialize_parameters(layer_dims, seed=42):
    np.random.seed(seed)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "relu":
        A = relu(Z)
    cache = (linear_cache, Z)
    return A, cache

def L_model_forward(X, parameters, use_relu=True):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation="relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="sigmoid")
    caches.append(cache)
    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (1./m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))
    return np.squeeze(cost)

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, Z = cache
    if activation == "relu":
        dZ = dA * (Z > 0)
    elif activation == "sigmoid":
        s = sigmoid(Z)
        dZ = dA * s * (1 - s)
    return linear_backward(dZ, linear_cache)

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, caches[-1], activation="sigmoid")
    for l in reversed(range(L-1)):
        grads["dA" + str(l)], grads["dW" + str(l+1)], grads["db" + str(l+1)] = linear_activation_backward(
            grads["dA" + str(l+1)], caches[l], activation="relu")
    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters['W' + str(l+1)] -= learning_rate * grads['dW' + str(l+1)]
        parameters['b' + str(l+1)] -= learning_rate * grads['db' + str(l+1)]
    return parameters

def correlation_matrix(df):
    cols = df.columns
    n = len(cols)
    cor_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cor_matrix[i, j] = np.corrcoef(df[cols[i]], df[cols[j]])[0, 1]
    return pd.DataFrame(cor_matrix, index=cols, columns=cols)

# --- Load and preprocess dataset ---
try:
    df = pd.read_csv('student_habits_performance.csv')
    df = df.drop('student_id', axis=1)
except FileNotFoundError:
    st.error("CSV file not found. Please ensure it's uploaded.")
    st.stop()

# --- Data cleaning ---
df = df.dropna()
if 'gender' in df.columns and 'Other' in df['gender'].unique():
    df['gender'] = df['gender'].replace('Other', df['gender'].mode()[0])

# Label encode gender
label_encoder = LabelEncoder()
df['gender'] = label_encoder.fit_transform(df['gender'])

# One-hot encode other categorical variables
df = pd.get_dummies(df, columns=[
    'part_time_job', 'diet_quality', 'parental_education_level',
    'internet_quality', 'extracurricular_participation'
], drop_first=True)

# Normalize numeric columns
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# --- Split features and labels ---
X = df.drop('exam_score', axis=1).values.T
y = df['exam_score'].values.reshape(1, -1)

# --- Train the model ---
layer_dims = [X.shape[0], 20, 7, 1]
parameters = initialize_parameters(layer_dims)
learning_rate = 0.009
num_iterations = 1000

costs = []
for i in range(num_iterations):
    AL, caches = L_model_forward(X, parameters)
    cost = compute_cost(AL, y)
    grads = L_model_backward(AL, y, caches)
    parameters = update_parameters(parameters, grads, learning_rate)
    if i % 100 == 0:
        costs.append(cost)

# --- Streamlit UI ---
st.title("🎓 Student Performance Prediction")

st.sidebar.header("📊 Input Student Habits")

# Get input data
input_data = {}
input_data['age'] = st.sidebar.slider('Age', 17, 25, 20)
input_data['gender'] = st.sidebar.radio('Gender', ['Female', 'Male'])  # Will be label-encoded
input_data['study_hours_per_day'] = st.sidebar.slider('Study Hours/Day', 0.0, 10.0, 3.0)
input_data['social_media_hours'] = st.sidebar.slider('Social Media Hours', 0.0, 10.0, 2.0)
input_data['netflix_hours'] = st.sidebar.slider('Netflix Hours', 0.0, 10.0, 1.0)
input_data['part_time_job_Yes'] = st.sidebar.radio('Part-Time Job?', [0, 1])
input_data['attendance_percentage'] = st.sidebar.slider('Attendance %', 0, 100, 85)
input_data['sleep_hours'] = st.sidebar.slider('Sleep Hours', 0.0, 10.0, 7.0)
input_data['diet_quality_Good'] = st.sidebar.radio('Diet Quality Good?', [0, 1])
input_data['exercise_frequency'] = st.sidebar.slider('Exercise Frequency (days/week)', 0, 7, 3)
input_data['parental_education_level_High School'] = st.sidebar.radio('Parental Education = High School?', [0, 1])
input_data['internet_quality_Good'] = st.sidebar.radio('Good Internet?', [0, 1])
input_data['mental_health_rating'] = st.sidebar.slider('Mental Health Rating (1-10)', 1, 10, 5)
input_data['extracurricular_participation_Yes'] = st.sidebar.radio('Extracurriculars?', [0, 1])

# Prepare input dataframe
input_df = pd.DataFrame([input_data])
input_df['gender'] = label_encoder.transform([input_data['gender']])[0]

# Reorder columns to match training
input_df = input_df[df.drop('exam_score', axis=1).columns]

# Scale input
input_scaled = scaler.transform(input_df).T

# --- Predict ---
predicted_score, _ = L_model_forward(input_scaled, parameters)
predicted_score = predicted_score[0][0] * 100  # Scale back to 0–100

st.subheader("📈 Predicted Exam Score")
st.success(f"🎯 **{predicted_score:.2f} / 100**")

# --- Visualizations ---
st.subheader("🔍 Correlation Heatmap")
cor_df = correlation_matrix(df[numerical_cols])
cor_data = cor_df.stack().reset_index(name='correlation')
cor_data.columns = ['Feature 1', 'Feature 2', 'correlation']
heatmap = alt.Chart(cor_data).mark_rect().encode(
    x='Feature 1:O',
    y='Feature 2:O',
    color='correlation:Q'
).properties(width=600, height=500)
st.altair_chart(heatmap, use_container_width=True)

# Feature importance based on weight magnitude
try:
    importances = abs(parameters['W1']).mean(axis=1)
    feature_importance = pd.DataFrame({
        'Feature': df.drop('exam_score', axis=1).columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    st.subheader("🔥 Feature Importance")
    bar_chart = alt.Chart(feature_importance).mark_bar().encode(
        x=alt.X('Importance:Q'),
        y=alt.Y('Feature:N', sort='-x'),
        tooltip=['Feature', 'Importance']
    ).properties(width=600)
    st.altair_chart(bar_chart, use_container_width=True)
except Exception as e:
    st.warning(f"Could not render feature importance: {e}")
