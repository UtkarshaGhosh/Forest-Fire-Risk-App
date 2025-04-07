# app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Load Dataset
@st.cache_data
def load_data():
    url = "forestfires.csv"
    df = pd.read_csv(url)
    return df

# 2. Preprocess Data
def preprocess_data(df):
    df = df.copy()
    month_map = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
                 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
    day_map = {'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5, 'sat':6, 'sun':7}
    df['month'] = df['month'].map(month_map)
    df['day'] = df['day'].map(day_map)
    df['fire_risk'] = df['area'].apply(lambda x: 1 if x > 0 else 0)
    X = df.drop(['area', 'fire_risk'], axis=1)
    y = df['fire_risk']
    return X, y, df

# 3. Train Model
def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# 4. App Title
st.title("🔥 Forest Fire Risk Predictor")

# Load data and train model
data = load_data()
X, y, full_df = preprocess_data(data)
model = train_model(X, y)

# Sidebar inputs
st.sidebar.header("Input Parameters")
def user_input_features():
    X_val = st.sidebar.slider('X Coordinate', 1, 9, 4)
    Y_val = st.sidebar.slider('Y Coordinate', 2, 9, 4)
    FFMC = st.sidebar.slider('FFMC Index', 0.0, 100.0, 85.0)
    DMC = st.sidebar.slider('DMC Index', 0.0, 100.0, 26.0)
    DC = st.sidebar.slider('DC Index', 0.0, 300.0, 94.3)
    ISI = st.sidebar.slider('ISI Index', 0.0, 20.0, 5.1)
    temp = st.sidebar.slider('Temperature (°C)', 0.0, 50.0, 18.0)
    RH = st.sidebar.slider('Relative Humidity (%)', 10, 100, 42)
    wind = st.sidebar.slider('Wind Speed (km/h)', 0.0, 20.0, 4.4)
    rain = st.sidebar.slider('Rainfall (mm)', 0.0, 10.0, 0.0)
    month = st.sidebar.slider('Month (1=Jan, 12=Dec)', 1, 12, 8)
    day = st.sidebar.slider('Day (1=Mon, 7=Sun)', 1, 7, 4)

    X_input = pd.DataFrame([[X_val, Y_val, month, day, FFMC, DMC, DC, ISI, temp, RH, wind, rain]],
                           columns=['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain'])
    return X_input

# Get prediction
input_df = user_input_features()
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

# Display output
if prediction == 1:
    st.error(f"⚠️ High Fire Risk Detected! Probability: {probability*100:.2f}%")
else:
    st.success(f"✅ Low Fire Risk. Probability: {probability*100:.2f}%")

# Display heatmap
st.subheader("🔥 Feature Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(full_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)
