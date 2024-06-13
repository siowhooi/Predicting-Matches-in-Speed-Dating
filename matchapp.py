import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Define gender mapping
gender_mapping = {
    'Female': 0,
    'Male': 1
}

# Define goal mapping
goal_mapping = {
    'Seemed like a fun night out': 1,
    'To meet new people': 2,
    'To get a date': 3,
    'Looking for a serious relationship': 4,
    'To say I did it': 5,
    'Other': 6
}

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
    inputs['gender'] = st.sidebar.selectbox('Select your gender', list(gender_mapping.keys()))
    inputs['age'] = st.sidebar.slider('Select your age', 18, 60, 30)
    inputs['income'] = st.sidebar.slider('Select your income', 0, 200000, 50000, step=1000)
    
    inputs['goal'] = st.sidebar.selectbox('Select your primary goal', list(goal_mapping.keys()))
    
    inputs['attr'] = st.sidebar.slider('Rate the opposite sex\'s attractiveness (1-10)', 1, 10, 5)
    inputs['sinc'] = st.sidebar.slider('Rate the opposite sex\'s sincerity (1-10)', 1, 10, 5)
    inputs['intel'] = st.sidebar.slider('Rate the opposite sex\'s intelligence (1-10)', 1, 10, 5)
    inputs['fun'] = st.sidebar.slider('Rate the opposite sex\'s fun (1-10)', 1, 10, 5)
    inputs['amb'] = st.sidebar.slider('Rate the opposite sex\'s ambitiousness (1-10)', 1, 10, 5)
    inputs['shar'] = st.sidebar.slider('Rate the opposite sex\'s shared interests (1-10)', 1, 10, 5)
    inputs['like'] = st.sidebar.slider('Overall, how much do you like this person? (1-10)', 1, 10, 5)
    
    # New features
    inputs['prob'] = st.sidebar.slider('How probable do you think it is that this person will say \'yes\' for you? (1-10)', 1, 10, 5)
    inputs['met'] = st.sidebar.selectbox('Have you met this person before?', ['Yes', 'No'])
    inputs['met'] = 1 if inputs['met'] == 'Yes' else 0
    
    return pd.DataFrame(inputs, index=[0])

input_df = user_input_features()

# Ensure input_df has the same columns as X_train
missing_cols = set(X_train.columns) - set(input_df.columns)
for col in missing_cols:
    input_df[col] = 0  # Assuming default values for missing columns

# Map gender and goal to numerical values
input_df['gender'] = input_df['gender'].map(gender_mapping)
input_df['goal'] = input_df['goal'].map(goal_mapping)

# Reorder input_df columns to match X_train columns
input_df = input_df[X_train.columns]

# Load the model
model = joblib.load('rf_classifier_model.pkl')

# Prediction
prediction = model.predict(input_df)

st.subheader('Prediction:')
st.write("It's a Match!" if prediction[0] == 1 else 'Not a Match.')
