import streamlit as st
import pickle
import pandas as pd

# Load the trained model from a file
with open('matches_predicting_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the Streamlit app
st.title("Predicting Matches in Speed Dating ")

# Input fields for the features
gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
age = st.number_input("Age", min_value=18, max_value=99, value=25)
income = st.number_input("Income", min_value=0, value=50000)
goal = st.selectbox("Goal", options=[1, 2, 3, 4, 5, 6], format_func=lambda x: {
    1: "Seemed like a fun night out",
    2: "To meet new people",
    3: "To get a date",
    4: "Looking for a serious relationship",
    5: "To say I did it",
    6: "Other"
}[x])

attr = st.slider("Attractiveness", min_value=1, max_value=10, value=5)
sinc = st.slider("Sincerity", min_value=1, max_value=10, value=5)
intel = st.slider("Intelligence", min_value=1, max_value=10, value=5)
fun = st.slider("Fun", min_value=1, max_value=10, value=5)
amb = st.slider("Ambitiousness", min_value=1, max_value=10, value=5)
shar = st.slider("Shared Interests", min_value=1, max_value=10, value=5)
like = st.slider("Overall Rating", min_value=1, max_value=10, value=5)
prob = st.slider("Probability of Interest Reciprocation", min_value=1, max_value=10, value=5)
met = st.selectbox("Met Before", options=[1, 2], format_func=lambda x: "Yes" if x == 1 else "No")

# Create a DataFrame with the input values
input_data = pd.DataFrame({
    'gender': [gender],
    'age': [age],
    'income': [income],
    'goal': [goal],
    'attr': [attr],
    'sinc': [sinc],
    'intel': [intel],
    'fun': [fun],
    'amb': [amb],
    'shar': [shar],
    'like': [like],
    'prob': [prob],
    'met': [met]
})

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("It's a Match!")
    else:
        st.warning("Not a Match.")

