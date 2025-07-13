# app/app.py

import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load('../models/qb_td_model.pkl')

st.title("🏈 NFL QB Touchdown Predictor")
st.subheader("Will this quarterback throw a touchdown in the next game?")

st.markdown("Fill out the quarterback's recent performance and bio details.")

# User input
passing_yards = st.number_input("Avg Passing Yards (last 3 games)", min_value=0, value=250)
td_passes = st.number_input("Avg TD Passes (last 3 games)", min_value=0.0, value=1.0)
pass_attempts = st.number_input("Avg Pass Attempts (last 3 games)", min_value=0.0, value=30.0)
age = st.number_input("Player Age", min_value=18, value=28)
experience = st.number_input("NFL Seasons Played", min_value=0, value=4)
height = st.number_input("Height (inches)", min_value=60.0, value=74.0)
weight = st.number_input("Weight (lbs)", min_value=150.0, value=220.0)

if st.button(" Predict Touchdown"):
    input_features = np.array([[passing_yards, td_passes, pass_attempts, age, experience, height, weight]])
    prediction = model.predict(input_features)[0]
    prob = model.predict_proba(input_features)[0][1]

    if prediction == 1:
        st.success(f" Likely to throw a TD! (Confidence: {prob:.2%})")
    else:
        st.error(f"Unlikely to throw a TD. (Confidence: {1 - prob:.2%})")

st.markdown("---")
st.caption("Built with 💡 by Shelton Bumhe")
