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
    X_coord = st.sidebar.slider('X Coordinate (1–9)', 1, 9, 4)
    Y_coord = st.sidebar.slider('Y Coordinate (2–9)', 2, 9, 4)

    X_input = pd.DataFrame([[FFMC, DMC, DC, ISI, temp, RH, wind, rain, month, day, X_coord, Y_coord]],
                           columns=['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'month', 'day', 'X', 'Y'])
    return X_input

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

# Show sample data
st.subheader("📊 Sample of the Dataset")
st.dataframe(data.head(10))

# Interactive Fire Map
st.subheader("🗌️ Fire Incident Map (Simulated Location)")

# Simulate lat/lon from X/Y grid
map_df = full_df[['X', 'Y', 'fire_risk']].copy()
map_df = map_df.rename(columns={"X": "lon", "Y": "lat"})
map_df["lat"] = 40.0 + map_df["lat"] * 0.01
map_df["lon"] = -8.0 + map_df["lon"] * 0.01

# Filter only fire incidents
fire_locations = map_df[map_df["fire_risk"] == 1][["lat", "lon"]]

# Plot map
st.map(fire_locations)

# Fire Risk Factor Explanation
st.subheader("📘 How Fire Risk Depends on Each Factor")
st.markdown("""
- **FFMC (Fine Fuel Moisture Code)**: Indicates how dry fine fuels are. Higher FFMC means fuels ignite easily.
- **DMC (Duff Moisture Code)**: Reflects moisture in loosely compacted organic layers. Higher DMC suggests increased fire risk.
- **DC (Drought Code)**: Measures long-term drying. High values indicate dry deep soil layers which contribute to persistent fires.
- **ISI (Initial Spread Index)**: Combines wind and FFMC to estimate fire spread. High ISI means fast spread.
- **Temperature**: Higher temps dry out fuel faster, increasing fire risk.
- **Relative Humidity (RH)**: Lower humidity means drier air and fuel, raising fire risk.
- **Wind Speed**: Stronger winds help fires spread quickly.
- **Rainfall**: More rain reduces fire likelihood by increasing fuel moisture.
- **Month & Day**: Seasonal and weekly trends impact conditions and human activities.
- **X & Y Coordinates**: Indicate location of fire-prone zones in the grid. Some areas may historically show higher fire risk due to vegetation, terrain, or human activity.
""")
