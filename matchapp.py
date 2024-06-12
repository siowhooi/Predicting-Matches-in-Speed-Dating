{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "0d05b1bb-54c0-47ff-b4fe-ffc475cf3ea7",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.model_selection import train_test_split\n",
    "import numpy as np\n",
    "\n",
    "# Title\n",
    "st.title('Speed Dating Match Predictor')\n",
    "\n",
    "# Description\n",
    "st.write(\"\"\"\n",
    "This app predicts the success of speed dating matches using a Random Forest Classifier.\n",
    "\"\"\")\n",
    "\n",
    "# Load and preprocess data\n",
    "def load_data():\n",
    "    df = pd.read_csv('speed_data_data.csv')\n",
    "    df.dropna(inplace=True)\n",
    "    df = df.drop('career', axis=1)\n",
    "    return df\n",
    "\n",
    "df = load_data()\n",
    "\n",
    "# Separate features and target\n",
    "X = df.drop('dec', axis=1)\n",
    "y = df['dec']\n",
    "\n",
    "# Split the data into training and testing sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Initialize and train the Random Forest Classifier\n",
    "rf_classifier = RandomForestClassifier(random_state=42)\n",
    "rf_classifier.fit(X_train, y_train)\n",
    "\n",
    "# Get user input\n",
    "st.sidebar.header('User Input Parameters')\n",
    "def user_input_features():\n",
    "    data = {}\n",
    "    for feature in X.columns:\n",
    "        data[feature] = st.sidebar.number_input(feature, min_value=float(df[feature].min()), max_value=float(df[feature].max()), value=float(df[feature].mean()))\n",
    "    features = pd.DataFrame(data, index=[0])\n",
    "    return features\n",
    "\n",
    "input_df = user_input_features()\n",
    "\n",
    "# Predict on the user input\n",
    "prediction = rf_classifier.predict(input_df)\n",
    "prediction_proba = rf_classifier.predict_proba(input_df)\n",
    "\n",
    "# Display results\n",
    "st.subheader('Prediction')\n",
    "st.write('It is a Match!' if prediction[0] == 1 else 'No a Match...')\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f926f40d-8a79-49e5-9232-e3976bb778f9",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8572cf6e-199b-4734-b48e-11354bb91e4f",
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
