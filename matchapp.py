import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load data
@st.cache  # Cache the data loading to optimize performance
def load_data():
    df = pd.read_csv('speed_data_data.csv')
    # Drop rows with missing values
    df.dropna(inplace=True)
    return df

# Load the data
df = load_data()

# Drop the 'career' column if it exists
if 'career' in df.columns:
    df = df.drop('career', axis=1)

# Separate features and target
X = df.drop('dec', axis=1)
y = df['dec']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Save the model
joblib.dump(rf_classifier, 'rf_classifier_model.pkl')

# Streamlit app starts here
st.title('Speed Dating Match Predictor')

# User input features
st.sidebar.header('User Input Features')

def user_input_features():
    inputs = {}
    for col in X.columns:
        inputs[col] = st.sidebar.number_input(f'Enter {col}', min_value=0.0, max_value=10.0, step=0.1)
    return pd.DataFrame(inputs, index=[0])

input_df = user_input_features()

# Load the model
model = joblib.load('rf_classifier_model.pkl')

# Prediction
prediction = model.predict(input_df)

st.subheader('Prediction')
st.write("It's a Match!" if prediction[0] == 1 else 'Not a Match.')
