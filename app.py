# app.py
pip install plotly
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import plotly.figure_factory as ff

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

    input_dict = {
        'X': X_coord,
        'Y': Y_coord,
        'month': month,
        'day': day,
        'FFMC': FFMC,
        'DMC': DMC,
        'DC': DC,
        'ISI': ISI,
        'temp': temp,
        'RH': RH,
        'wind': wind,
        'rain': rain,
    }
    X_input = pd.DataFrame([input_dict])
    X_input = pd.concat([X[X.columns].iloc[:0], X_input], ignore_index=True)
    return X_input

input_df = user_input_features()
if st.sidebar.button("Submit Prediction"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("⚠️ High Fire Risk Detected!")
    else:
        st.success("✅ Low Fire Risk.")

# ➕ Custom input section
st.subheader("🧪 Custom Prediction Test")
with st.expander("Test With Specific Values"):
    col1, col2 = st.columns(2)
    with col1:
        ffmc_val = st.number_input("FFMC", value=92.78)
        dmc_val = st.number_input("DMC", value=87.46)
        dc_val = st.number_input("DC", value=250.71)
        isi_val = st.number_input("ISI", value=16.64)
        temp_val = st.number_input("Temperature", value=40.37)
    with col2:
        rh_val = st.number_input("Relative Humidity", value=22)
        wind_val = st.number_input("Wind Speed", value=4.4)
        rain_val = st.number_input("Rainfall", value=2.51)
        month_val = st.number_input("Month", value=9)
        day_val = st.number_input("Day", value=3)

    if st.button("Predict Fire Risk for Custom Input"):
        custom_input = pd.DataFrame([{ 
            'X': 4, 'Y': 4, 'month': month_val, 'day': day_val,
            'FFMC': ffmc_val, 'DMC': dmc_val, 'DC': dc_val, 'ISI': isi_val,
            'temp': temp_val, 'RH': rh_val, 'wind': wind_val, 'rain': rain_val
        }])
        custom_input = pd.concat([X[X.columns].iloc[:0], custom_input], ignore_index=True)
        pred = model.predict(custom_input)[0]
        if pred == 1:
            st.error("🔥 Fire Risk: HIGH")
        else:
            st.success("🌲 Fire Risk: LOW")

# Display heatmap
st.subheader("🔥 Feature Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(full_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Show sample data
st.subheader("📊 Sample of the Dataset")
st.dataframe(data.head(10))

# Interactive Fire Map
st.subheader("🗜️ Fire Incident Map (Simulated Location)")

map_df = full_df[['X', 'Y', 'fire_risk']].copy()
map_df = map_df.rename(columns={"X": "lon", "Y": "lat"})
map_df["lat"] = 40.0 + map_df["lat"] * 0.01
map_df["lon"] = -8.0 + map_df["lon"] * 0.01
fire_locations = map_df[map_df["fire_risk"] == 1][["lat", "lon"]]

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

# Classification Report and Confusion Matrix
st.subheader("🧮 Model Evaluation")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model_eval = RandomForestClassifier(n_estimators=100, random_state=42)
model_eval.fit(X_train, y_train)
y_pred = model_eval.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
labels = ["No Fire", "Fire"]
cm_fig = ff.create_annotated_heatmap(
    z=cm,
    x=labels,
    y=labels,
    colorscale='Blues',
    showscale=True,
    annotation_text=[[str(cell) for cell in row] for row in cm]
)
st.plotly_chart(cm_fig, use_container_width=True)

report = classification_report(y_test, y_pred, target_names=labels, output_dict=False)
st.text("📋 Classification Report")
st.code(report, language='text')
