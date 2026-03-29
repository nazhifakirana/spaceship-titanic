import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/logreg_pipeline.pkl")

st.title("ASG 04 MD - Nazhifa - Spaceship Titanic Model Deployment")

HomePlanet = st.selectbox("HomePlanet", ["Earth","Europa","Mars"])
CryoSleep = st.selectbox("CryoSleep", [True, False])
Destination = st.selectbox("Destination", ["TRAPPIST-1e","PSO J318.5-22","55 Cancri e"])
Age = st.number_input("Age", value=30)
VIP = st.selectbox("VIP", [True, False])

RoomService = st.number_input("RoomService", value=0)
FoodCourt = st.number_input("FoodCourt", value=0)
ShoppingMall = st.number_input("ShoppingMall", value=0)
Spa = st.number_input("Spa", value=0)
VRDeck = st.number_input("VRDeck", value=0)

Deck = st.selectbox("Deck", ["A","B","C","D","E","F","G","T"])
Cabin_num = st.number_input("Cabin_num", value=0)
Side = st.selectbox("Side", ["P","S"])

input_data = pd.DataFrame({
    "PassengerId": ["0001_01"],
    "HomePlanet": [HomePlanet],
    "CryoSleep": [CryoSleep],
    "Cabin": [f"{Deck}/{Cabin_num}/{Side}"],
    "Destination": [Destination],
    "Age": [Age],
    "VIP": [VIP],
    "RoomService": [RoomService],
    "FoodCourt": [FoodCourt],
    "ShoppingMall": [ShoppingMall],
    "Spa": [Spa],
    "VRDeck": [VRDeck],
    "Name": ["John Doe"]
})

if st.button("Predict"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Passenger was Transported")
    else:
        st.error("Passenger was NOT Transported")
