{
 "cells": [
  {
   "cell_type": "code",
   
   "id": "5cc7bd8c-a23b-4f1c-bcb4-a6d24b313aed",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.model_selection import train_test_split\n",
    "import joblib\n",
    "\n",
    "# Load data\n",
    "def load_data():\n",
    "    df = pd.read_csv('cleaned_speed_data.csv')\n",
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
    "# Train the model\n",
    "rf_classifier = RandomForestClassifier(random_state=42)\n",
    "rf_classifier.fit(X_train, y_train)\n",
    "\n",
    "# Save the model\n",
    "joblib.dump(rf_classifier, 'rf_classifier_model.pkl')\n",
    "\n",
    "st.title('Speed Dating Match Predictor')\n",
    "\n",
    "# User input features\n",
    "st.sidebar.header('User Input Features')\n",
    "\n",
    "def user_input_features():\n",
    "    inputs = {}\n",
    "    for col in X.columns:\n",
    "        inputs[col] = st.sidebar.number_input(f'Enter {col}', min_value=0.0, max_value=10.0, step=0.1)\n",
    "    return pd.DataFrame(inputs, index=[0])\n",
    "\n",
    "input_df = user_input_features()\n",
    "\n",
    "# Load the model\n",
    "model = joblib.load('rf_classifier_model.pkl')\n",
    "\n",
    "# Prediction\n",
    "prediction = model.predict(input_df)\n",
    "\n",
    "st.subheader('Prediction')\n",
    "st.write(\"It's a Match!\" if prediction[0] == 1 else 'Not a Match.')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2c17868d-2bac-47a0-979f-088ed8887ef3",
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
