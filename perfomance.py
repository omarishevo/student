import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
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

# --- Data Cleaning ---
# Handle missing values
df = df.dropna()

# Handle outliers (example: clipping extreme values)
for col in df.select_dtypes(include=np.number).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower_bound, upper_bound)

# --- Preprocessing ---
# Convert gender to numerical (Label Encoding)
label_encoder = LabelEncoder()
df['gender'] = label_encoder.fit_transform(df['gender'])

# Convert other categorical features to numerical (One-Hot Encoding)
df = pd.get_dummies(df, columns=['part_time_job', 'diet_quality', 'parental_education_level', 'internet_quality', 'extracurricular_participation'], drop_first=True)

# Normalize numerical features
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Separate features (X) and target (y)
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

# Create input widgets for each feature
input_dict = {}
input_dict['age'] = st.sidebar.slider('Age', 17, 25, 20)
input_dict['gender'] = st.sidebar.radio('Gender', ['Female', 'Male'], index=0)
input_dict['study_hours_per_day'] = st.sidebar.slider('Study Hours per Day', 0.0, 8.0, 4.0)
input_dict['social_media_hours'] = st.sidebar.slider('Social Media Hours', 0.0, 8.0, 2.0)
input_dict['netflix_hours'] = st.sidebar.slider('Netflix Hours', 0.0, 8.0, 1.0)
input_dict['part_time_job_Yes'] = st.sidebar.radio('Part-time Job', [0, 1], index=0, format_func=lambda x: "No" if x == 0 else "Yes")
input_dict['attendance_percentage'] = st.sidebar.slider('Attendance Percentage', 0, 100, 80)
input_dict['sleep_hours'] = st.sidebar.slider('Sleep Hours', 0.0, 10.0, 7.0)
input_dict['diet_quality_Good'] = st.sidebar.radio('Diet Quality', [0, 1], index=0, format_func=lambda x: "Fair/Poor" if x == 0 else "Good")
input_dict['exercise_frequency'] = st.sidebar.slider('Exercise Frequency', 0, 7, 3)
input_dict['parental_education_level_High School'] = st.sidebar.radio('Parental Education Level', [0, 1], index=0, format_func=lambda x: "None/Other" if x == 0 else "High School")
input_dict['internet_quality_Good'] = st.sidebar.radio('Internet Quality', [0, 1], index=0, format_func=lambda x: "Average/Poor" if x == 0 else "Good")
input_dict['mental_health_rating'] = st.sidebar.slider('Mental Health Rating', 1, 10, 5)
input_dict['extracurricular_participation_Yes'] = st.sidebar.radio('Extracurricular Participation', [0, 1], index=0, format_func=lambda x: "No" if x == 0 else "Yes")

# --- Prediction ---
# Create a DataFrame from the input values
input_data = pd.DataFrame([input_dict])

# Encode 'gender'
input_data['gender'] = label_encoder.transform(input_data['gender'])

# Scale the input data
input_scaled = scaler.transform(input_data)
input_scaled = input_scaled.T

# Make Prediction
AL, caches = L_model_forward(input_scaled, parameters)

# Display the prediction
st.subheader('Prediction')
st.markdown(f'<p class="big-font">Predicted Exam Score: {AL[0][0]:.2f}</p>', unsafe_allow_html=True)

# --- Visualizations ---
st.subheader('Visualizations')

# Correlation Heatmap using Altair
st.write("Correlation Heatmap:")
corr = df.corr()
corr = corr.stack().reset_index(name="correlation")

heatmap = alt.Chart(corr).mark_rect().encode(
    x=alt.X('level_0:O', title='Variable 1'),
    y=alt.Y('level_1:O', title='Variable 2'),
    color=alt.Color('correlation:Q', scale=alt.Scale(scheme='viridis'))
).properties(
    width=600,
    height=400,
    title='Correlation Heatmap'
)
st.altair_chart(heatmap, use_container_width=True)

# Feature Importance using Altair
st.write("Feature Importance:")
try:
    feature_importance = pd.DataFrame({'Feature': df.drop('exam_score', axis=1).columns, 'Importance': abs(parameters['W1']).mean(axis=0)})
    feature_importance = feature_importance.sort_values('Importance', ascending=False)

    barplot = alt.Chart(feature_importance).mark_bar().encode(
        x=alt.X('Feature:O', title='Feature'),
        y=alt.Y('Importance:Q', title='Importance'),
        tooltip=['Feature', 'Importance']
    ).properties(
        width=600,
        height=400,
        title='Feature Importance'
    )
    st.altair_chart(barplot, use_container_width=True)

except Exception as e:
        st.error(f"Error generating feature importance plot: {e}")
