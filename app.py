import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

st.title("House Price Prediction")

# Load the trained Ridge Regression model
with open('ridge_model.pkl', 'rb') as model_file:
    ridge_model = pickle.load(model_file)

# Load your dataset (assuming you have it in a CSV file)
# Replace 'your_dataset.csv' with the actual file path of your dataset
df = pd.read_csv('Cleaned_data.csv')

# Extract unique locations from your dataset
unique_locations = df['location'].unique()

# Create an instance of the OneHotEncoder
encoder = OneHotEncoder(sparse=False)

# Fit the encoder on the unique locations
encoder.fit(unique_locations.reshape(-1, 1))

# Create input fields for user interaction
st.sidebar.header('Enter Property Details')
area = st.sidebar.number_input('Area (sq. ft.)', min_value=1)
bedrooms = st.sidebar.number_input('Bedrooms', min_value=1,max_value=16)
bathrooms = st.sidebar.number_input('Bathrooms', min_value=1,max_value=16)
location = st.sidebar.selectbox('Location', unique_locations)



# Create a button to trigger prediction
if st.sidebar.button('Predict Price'):
    # Create a feature array with the same order as used during training
    input_data = np.array([[area, bedrooms, bathrooms]])

    # One-hot encode the selected location
    selected_location_encoded = encoder.transform(np.array([[location]]))

    # Concatenate the one-hot encoded location with the other features
    input_data = np.concatenate([input_data, selected_location_encoded], axis=1)

    # Make a prediction using the loaded Ridge model
    prediction = ridge_model.predict(input_data)

    # Display the predicted price
    # Display the predicted price in lakhs (Lac)
    st.subheader(f'Predicted Price: ${prediction[0]/ 100:.2f} Lac')
    # Sidebar for user input
    st.sidebar.header("Enter Property Details")

# Add input fields and prediction code here


# Display instructions or descriptions
# Larger gap using multiple <br> tags
st.markdown("<br><br><br>", unsafe_allow_html=True)

# Instructions
st.write("This is a simple house price prediction app.")
st.write("Please enter the property details on the left sidebar and click the 'Predict' button.")
st.write("The predicted price will be displayed above based on the entered details.")
