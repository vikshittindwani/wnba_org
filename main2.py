import streamlit as st
import pandas as pd
import pickle

# Load model, encoder, and feature list
model = pickle.load(open('pipe.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
model_features = pickle.load(open('model_features.pkl', 'rb'))

# Teams used
teams = [
    'Atlanta Dream', 'Chicago Sky', 'Connecticut Sun', 'Dallas Wings',
    'Indiana Fever', 'Las Vegas Aces', 'Los Angeles Sparks', 'Minnesota Lynx',
    'New York Liberty', 'Phoenix Mercury', 'Seattle Storm', 'Washington Mystics'
]

st.title("🏀 WNBA Winner Predictor (Team-Based)")

# User selects only teams
away_team = st.selectbox("Away Team", teams)
home_team = st.selectbox("Home Team", [team for team in teams if team != away_team])

if st.button("Predict Winner"):
    # Set up base input
    input_dict = {feature: 0 for feature in model_features}
    input_dict['Away Score'] = 0
    input_dict['Home Score'] = 0
    input_dict[f'Away Team_{away_team}'] = 1
    input_dict[f'Home Team_{home_team}'] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])[model_features]

    # Predict
    pred = model.predict(input_df)[0]
    winner = label_encoder.inverse_transform([pred])[0]

    st.success(f"🎯 Predicted Winner: **{winner}**")
