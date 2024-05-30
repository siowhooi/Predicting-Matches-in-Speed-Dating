{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "5cc7bd8c-a23b-4f1c-bcb4-a6d24b313aed",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pickle\n",
    "import pandas as pd\n",
    "\n",
    "# Load the trained model from a file\n",
    "with open('matches_predicting_model.pkl', 'rb') as file:\n",
    "    model = pickle.load(file)\n",
    "\n",
    "# Define the Streamlit app\n",
    "st.title(\"Speed Dating Match Predictor\")\n",
    "\n",
    "# Input fields for the features\n",
    "gender = st.selectbox(\"Gender\", options=[0, 1], format_func=lambda x: \"Female\" if x == 0 else \"Male\")\n",
    "age = st.number_input(\"Age\", min_value=18, max_value=99, value=25)\n",
    "income = st.number_input(\"Income\", min_value=0, value=50000)\n",
    "goal = st.selectbox(\"Goal\", options=[1, 2, 3, 4, 5, 6], format_func=lambda x: {\n",
    "    1: \"Seemed like a fun night out\",\n",
    "    2: \"To meet new people\",\n",
    "    3: \"To get a date\",\n",
    "    4: \"Looking for a serious relationship\",\n",
    "    5: \"To say I did it\",\n",
    "    6: \"Other\"\n",
    "}[x])\n",
    "\n",
    "attr = st.slider(\"Attractiveness\", min_value=1, max_value=10, value=5)\n",
    "sinc = st.slider(\"Sincerity\", min_value=1, max_value=10, value=5)\n",
    "intel = st.slider(\"Intelligence\", min_value=1, max_value=10, value=5)\n",
    "fun = st.slider(\"Fun\", min_value=1, max_value=10, value=5)\n",
    "amb = st.slider(\"Ambitiousness\", min_value=1, max_value=10, value=5)\n",
    "shar = st.slider(\"Shared Interests\", min_value=1, max_value=10, value=5)\n",
    "like = st.slider(\"Overall Rating\", min_value=1, max_value=10, value=5)\n",
    "prob = st.slider(\"Probability of Interest Reciprocation\", min_value=1, max_value=10, value=5)\n",
    "met = st.selectbox(\"Met Before\", options=[1, 2], format_func=lambda x: \"Yes\" if x == 1 else \"No\")\n",
    "\n",
    "# Create a DataFrame with the input values\n",
    "input_data = pd.DataFrame({\n",
    "    'gender': [gender],\n",
    "    'age': [age],\n",
    "    'income': [income],\n",
    "    'goal': [goal],\n",
    "    'attr': [attr],\n",
    "    'sinc': [sinc],\n",
    "    'intel': [intel],\n",
    "    'fun': [fun],\n",
    "    'amb': [amb],\n",
    "    'shar': [shar],\n",
    "    'like': [like],\n",
    "    'prob': [prob],\n",
    "    'met': [met]\n",
    "})\n",
    "\n",
    "# Make prediction\n",
    "if st.button(\"Predict\"):\n",
    "    prediction = model.predict(input_data)[0]\n",
    "    if prediction == 1:\n",
    "        st.success(\"It's a Match!\")\n",
    "    else:\n",
    "        st.warning(\"Not a Match.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7b13afec-a18f-4a89-b172-4ae9b1041144",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
