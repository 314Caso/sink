import streamlit as st
import pickle
import numpy as np

# Load Model & Scaler
model = pickle.load(open('titanic_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Streamlit UI
st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict survival.")

# User Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ["Male", "Female"])
age = st.slider("Age", 1, 80, 30)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
embarked = st.selectbox("Embarked Port", ["S", "C", "Q"])

# Convert Inputs
sex = 1 if sex == "Female" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

# Prepare Data for Model
user_data = np.array([[pclass, sex, age, sibsp, parch, embarked_Q, embarked_S]])
user_data_scaled = scaler.transform(user_data)

# Predict Survival
if st.button("Predict Survival"):
    prediction = model.predict(user_data_scaled)
    result = "YAAAAY!!! Dodged a bullet" if prediction[0] == 1 else "Ouch. RIP"
    st.success(f"Prediction: **{result}**")
