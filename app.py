# forest_fire_risk_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load Dataset
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/Forestfires.csv")
    return df

# 2. Preprocess Data
def preprocess_data(df):
    df = df.copy()
    month_map = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
                 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
    day_map = {'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5, 'sat':6, 'sun':7}
    df['month'] = df['month'].map(month_map)
    df['day'] = df['day'].map(day_map)
    df['fire_risk'] = df['area'].apply(lambda x: 1 if x > 0 else 0)  # Binary classification
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
    prediction = model.predict([input_data])
    probability = model.predict_proba([input_data])
    return prediction[0], probability[0][1]

# 5. Generate Risk Heatmap
def create_heatmap():
    m = folium.Map(location=[40.0, -3.0], zoom_start=5)

    # Dummy data points for demonstration
    fire_data = [
        {"lat": 40.0, "lon": -3.0, "risk": 0.9},
        {"lat": 41.0, "lon": -3.5, "risk": 0.6},
        {"lat": 39.5, "lon": -2.5, "risk": 0.2},
    ]

    for fire in fire_data:
        color = "red" if fire['risk'] > 0.75 else "orange" if fire['risk'] > 0.4 else "green"
        folium.CircleMarker(
            location=[fire['lat'], fire['lon']],
            radius=10,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            popup=f"Risk: {fire['risk']*100:.1f}%"
        ).add_to(m)

    return m

# 6. Main App UI
def main():
    st.title("🔥 Forest Fire Risk Predictor")

    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train)

    st.sidebar.header("Input Environmental Conditions")

    temp = st.sidebar.slider("Temperature (°C)", 0, 50, 25)
    RH = st.sidebar.slider("Relative Humidity (%)", 10, 100, 50)
    wind = st.sidebar.slider("Wind Speed (km/h)", 0, 20, 5)
    rain = st.sidebar.slider("Rainfall (mm)", 0.0, 10.0, 0.0)

    month = st.sidebar.selectbox("Month", list(range(1, 13)))
    day = st.sidebar.selectbox("Day of Week", list(range(1, 8)))

    FFMC = st.sidebar.slider("FFMC Index", 0.0, 100.0, 80.0)
    DMC = st.sidebar.slider("DMC Index", 0.0, 100.0, 30.0)
    DC = st.sidebar.slider("DC Index", 0.0, 300.0, 100.0)
    ISI = st.sidebar.slider("ISI Index", 0.0, 20.0, 10.0)

    input_data = [FFMC, DMC, DC, ISI, temp, RH, wind, rain, month, day]

    if st.button("Predict Fire Risk"):
        pred, prob = predict_risk(model, input_data)
        if pred == 1:
            st.error(f"⚠️ High Fire Risk Detected! Probability: {prob*100:.2f}%")
        else:
            st.success(f"✅ Low Fire Risk. Probability: {prob*100:.2f}%")

    if st.checkbox("Show Dataset"):
        st.write(df.head())

    if st.checkbox("Show Correlation Heatmap"):
        plt.figure(figsize=(10,6))
        sns.heatmap(df.corr(), annot=True, cmap='YlOrRd')
        st.pyplot(plt)

    if st.checkbox("Show Fire Risk Heatmap"):
        heatmap = create_heatmap()
        st_folium(heatmap, width=700)

if __name__ == '__main__':
    main()
