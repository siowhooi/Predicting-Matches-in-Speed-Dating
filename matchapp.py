{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e82b77bf-7d92-4d2f-8683-65c706c2686b",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "6e3408bb-4e1f-414e-9699-738e133aed2c",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-06-13 13:57:34.276 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\wohen\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score\n",
    "\n",
    "# Title of the web app\n",
    "st.title('Predicting A Speed Dating Match')\n",
    "\n",
    "# Load the dataset\n",
    "def load_data():\n",
    "    df = pd.read_csv('speed_data_data.csv')\n",
    "    df.dropna(inplace=True)\n",
    "    df = df.drop('career', axis=1)\n",
    "    return df\n",
    "\n",
    "df = load_data()\n",
    "\n",
    "# Display the data\n",
    "st.write(\"Data Overview\", df.head())\n",
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
    "# Predict on the testing set\n",
    "y_pred_rf = rf_classifier.predict(X_test)\n",
    "\n",
    "# Calculate accuracy\n",
    "accuracy = accuracy_score(y_test, y_pred_rf)\n",
    "\n",
    "# Display accuracy\n",
    "st.write(\"Model Accuracy\", accuracy)\n",
    "\n",
    "# User input for new predictions\n",
    "st.sidebar.header(\"User Input Features\")\n",
    "def user_input_features():\n",
    "    data = {}\n",
    "    for feature in X.columns:\n",
    "        data[feature] = st.sidebar.number_input(feature, float(X[feature].min()), float(X[feature].max()), float(X[feature].mean()))\n",
    "    features = pd.DataFrame(data, index=[0])\n",
    "    return features\n",
    "\n",
    "input_df = user_input_features()\n",
    "\n",
    "# Display user input\n",
    "st.write(\"User Input Features\", input_df)\n",
    "\n",
    "# Predict with user input\n",
    "prediction = rf_classifier.predict(input_df)\n",
    "st.write(\"Prediction\", \"Bravo! It is a Match!\" if prediction[0] == 1 else \"Not a Match\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7864fe9c-6bdf-4667-a3d8-607e87e4559d",
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
