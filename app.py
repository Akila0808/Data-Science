import streamlit as st
import pandas as pd
import joblib
import datetime

# Load the full pipeline (preprocessing + model)
model = joblib.load('cause_model.pkl')

st.set_page_config(page_title="Accident Cause Predictor", layout="centered")
st.title("🚗 Traffic Accident Cause Predictor")
st.markdown("Enter the accident details to predict the likely cause.")

# User Inputs
state = st.selectbox("State", ["State1", "State2", "State3"])  # Replace with real states
road_type = st.selectbox("Road Type", ["Highway", "Urban", "Rural"])
weather = st.selectbox("Weather Conditions", ["Clear", "Rainy", "Foggy", "Snowy"])
alcohol = st.selectbox("Alcohol Involved", ["Yes", "No"])
fatigue = st.selectbox("Driver Fatigue", ["Yes", "No"])
road_cond = st.selectbox("Road Conditions", ["Dry", "Wet", "Icy", "Gravel"])
speed_limit = st.number_input("Speed Limit (km/h)", min_value=20, max_value=150, step=5)
number_of_deaths = st.number_input("Number of Deaths", min_value=0, step=1)
number_of_injuries = st.number_input("Number of Injuries", min_value=0, step=1)

# Prepare input data (no need to encode/scale, the pipeline will handle it)
input_data = pd.DataFrame([{
    'State': state,
    'Road_Type': road_type,
    'Weather_Conditions': weather,
    'Alcohol_Involved': alcohol,
    'Driver_Fatigue': fatigue,
    'Road_Conditions': road_cond,
    'Speed_Limit': speed_limit,
    'Number_of_Deaths': number_of_deaths,
    'Number_of_Injuries': number_of_injuries
}])

if st.button("Predict Reason"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"🎯 Predicted Accident Cause: **{prediction}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
