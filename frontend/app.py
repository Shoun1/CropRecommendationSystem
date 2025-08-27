import streamlit as st
import joblib 
import numpy as np
import pandas as pd
import requests
tree_model = joblib.load('C:\\Users\\DELL\\SmartFarmingApp\\CropRecommendationSystem\\ML_Int\\models\\Tree_model.pkl')
bayesian_model = joblib.load('C:\\Users\\DELL\\SmartFarmingApp\\CropRecommendationSystem\\ML_Int\\models\\Bayesian_model.pkl')

st.title("Crop Recommendation System")

N = st.number_input("Nitrogen (N) in soil (kg/ha)", min_value=0, max_value=200, value=50)
P = st.number_input("Phosphorus (P) in soil (kg/ha)", min_value=0, max_value=200, value=50)
K = st.number_input("Potassium (K) in soil (kg/ha)", min_value=0, max_value=200, value=50)
ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0, max_value=3000, value=100)


if st.button("Predict"):
    '''input_data = np.array([[N, P, K, ph, rainfall]])
    crop_list = ['rice', 'maize', 'wheat', 'barley', 'sorghum', 'millet', 'groundnut', 'soybean', 'sunflower', 'cotton'
                    'sugarcane', 'tobacco', 'tea', 'coffee', 'coconut', 'banana', 'papaya', 'mango', 'orange']
    crop_labels = np.array(crop_list)

    # If you must decode a one-hot vector
    # predicted_index = np.argmax(tree_prediction)
    # st.write(f"Decision Tree Prediction: {crop_labels[predicted_index]}")
    # For simplicity, we assume the input data is already scaled
    #input_data_scaled = (input_data - np.array([50, 50, 50, 6.5, 100])) / np.array([50, 50, 50, 7.5, 2000])
    
    tree_prediction = tree_model.predict(input_data)
    predicted_index = np.argmax(tree_prediction)
    
    bayesian_prediction = bayesian_model.predict(input_data)

    st.write(f"Decision Tree Prediction: {crop_labels[predicted_index]}")
    st.write(f"Naive Bayes Prediction: {bayesian_prediction[0]}")'''

    response = requests.post("http://127.0.0.1:8000/predict",json={
        "N":N,
        "P":P,
        "K":K,
        "ph":ph,
        "rainfall":rainfall
    })

    if response.status_code == 200:
        st.success(f"Recommended Crop: {response.json()['prediction']}")

    else:
        st.error("Error occurred while fetching API prediction.")


