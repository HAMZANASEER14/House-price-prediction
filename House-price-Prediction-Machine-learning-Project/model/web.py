import streamlit as st
import pickle
import json
import numpy as np
from PIL import Image, ImageOps

# Load trained model
with open("house_price_prediction.pickle", "rb") as f:
    model = pickle.load(f)


# Load column names
with open("columns.json", "r") as f:
    columns = json.load(f)
    feature_columns = columns["data_columns"]

# Function to predict price
def predict_price(location, total_sqft, bath, bhk):
    # Create input array for prediction
    input_data = np.zeros(len(feature_columns))
    input_data[0] = total_sqft
    input_data[1] = bath
    input_data[2] = bhk
    try:
        loc_index = feature_columns.index(location.lower())
        input_data[loc_index] = 1
    except ValueError:
        pass  # Location not found in columns
    # Predict the price
    predicted_price = model.predict([input_data])[0]
    return predicted_price

# Streamlit UI
def main():
    # Load Image
    image = Image.open('../images/house.jpg')
    st.image(image, width=700)

    st.title("House Price Prediction")

    # Input components
    location = st.selectbox("Location", feature_columns[3:])  # Skip first 3 columns
    total_sqft = st.number_input("Total Square Feet Area")
    bath = st.number_input("Number of Bathrooms", min_value=1, step=1)
    bhk = st.number_input("Number of Bedrooms", min_value=1, step=1)

    # Real-time prediction
    predicted_price = predict_price(location, total_sqft, bath, bhk)
    # Convert to PKR (assuming model predicts in lakhs INR, convert to PKR)
    predicted_price_pkr = predicted_price * 320000  # 1 lakh INR ≈ 320,000 PKR
    st.success(f"Predicted Price: {predicted_price_pkr:,.0f} PKR")
if __name__ == "__main__":
    main()
