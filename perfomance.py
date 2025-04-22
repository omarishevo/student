import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import math

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
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="relu")
    caches.append(cache)

    AL = np.sum(AL, axis=0, keepdims=True)
    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    cost = np.squeeze(cost)
    return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = dA * (activation_cache > 0)
    elif activation == "sigmoid":
        s = sigmoid(activation_cache)
        dZ = dA * s * (1 - s)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation="relu")

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation="relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    return parameters

def predict(X, parameters):
    AL, caches = L_model_forward(X, parameters)
    predictions = (AL > 0.5)
    return predictions

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

# Normalize numerical features
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Split into features and target
X = df.drop('exam_score', axis=1).values.T
y = df['exam_score'].values.reshape(1, -1)

# --- Model Training ---
layer_dims = [X.shape[0], 20, 7, 1]  # Define the architecture
parameters = initialize_parameters(layer_dims, seed=42)
learning_rate = 0.009
num_iterations = 3000

costs = []
for i in range(0, num_iterations):
    AL, caches = L_model_forward(X, parameters)
    cost = compute_cost(AL, y)
    grads = L_model_backward(AL, y, caches)
    parameters = update_parameters(parameters, grads, learning_rate)

    if i % 100 == 0:
        costs.append(cost)
        print(f"Cost after iteration {i}: {cost}")

print("Training complete!")

# --- Streamlit App ---
st.markdown(
    """
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
    """,
    unsafe_allow_html=True
)

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

# Encode and Normalize input
for col in ['gender', 'part_time_job', 'diet_quality', 'parental_education_level', 'internet_quality', 'extracurricular_participation']:
    input_data[col] = LabelEncoder().fit_transform(input_data[col])

input_scaled = scaler.transform(input_data)
input_scaled = input_scaled.T # Transpose for the NN

# Make Prediction
AL, caches = L_model_forward(input_scaled, parameters)

# Display the prediction
st.subheader('Prediction')
st.markdown(f'<p class="big-font">Predicted Exam Score: {AL[0][0]:.2f}</p>', unsafe_allow_html=True)

# --- Visualizations ---
st.subheader('Visualizations')

# Correlation Heatmap
st.write("Correlation Heatmap:")
fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='viridis', ax=ax_corr)
st.pyplot(fig_corr)
