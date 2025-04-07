# forest_fire_risk_app_streamlit.py (Streamlit version)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Load Dataset
def load_data():
    df = pd.read_csv("forestfires.csv")
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
    return train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# 4. Predict Risk
def predict_risk(model, input_data):
    pred = model.predict([input_data])[0]
    prob = model.predict_proba([input_data])[0][1]
    if pred == 1:
        return f"\u26a0\ufe0f High Fire Risk Detected! Probability: {prob*100:.2f}%"
    else:
        return f"✅ Low Fire Risk. Probability: {prob*100:.2f}%"

# Load and Train
raw_data = load_data()
X_train, X_test, y_train, y_test = preprocess_data(raw_data)
model = train_model(X_train, y_train)

# Streamlit UI
st.set_page_config(page_title="Forest Fire Risk Predictor", layout="centered")
st.title("\ud83d\udd25 Forest Fire Risk Predictor")
st.write("Adjust the sliders based on current conditions to assess fire risk.")

FFMC = st.slider("FFMC Index", 0.0, 100.0, 80.0)
DMC = st.slider("DMC Index", 0.0, 100.0, 30.0)
DC = st.slider("DC Index", 0.0, 300.0, 100.0)
ISI = st.slider("ISI Index", 0.0, 20.0, 10.0)
temp = st.slider("Temperature (°C)", 0, 50, 25)
RH = st.slider("Relative Humidity (%)", 10, 100, 50)
wind = st.slider("Wind Speed (km/h)", 0, 20, 5)
rain = st.slider("Rainfall (mm)", 0.0, 10.0, 0.0)
month = st.slider("Month", 1, 12, 8)
day = st.slider("Day of Week", 1, 7, 4)

if st.button("Predict Fire Risk"):
    input_data = [FFMC, DMC, DC, ISI, temp, RH, wind, rain, month, day]
    result = predict_risk(model, input_data)
    st.success(result)
