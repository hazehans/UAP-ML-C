import pandas as pd
import streamlit as st
import joblib  # For loading the saved model

# Load the saved model
model = joblib.load("model/credit_fraud_model.pkl")

# Merchant input (using a selectbox or text input for more flexibility)
merchant = st.selectbox("Select Merchant", ["fraud_Rippin, Kub and Mann", "fraud_Heller, Gutmann and Zieme", "fraud_Kiehn Inc"])

# Category input (using a selectbox)
category = st.selectbox("Select Category", ["misc_net", "grocery_pos", "entertainment"])

# Gender input (using a selectbox)
gender = st.selectbox("Select Gender", ["M", "F"])

# City input (using a selectbox or text input)
city = st.selectbox("Select City", ["Moravian Falls", "Orient", "Malad City"])

# State input (using a selectbox)
state = st.selectbox("Select State", ["NC", "WA", "PA"])

# Job input (using a selectbox or text input)
job = st.selectbox("Select Job", ["Psychologist, counselling", "Patent attorney", "Pathologist"])

# Amount input (using a slider)
amt = st.slider("Transaction Amount", min_value=0, max_value=10000, step=1)

# City population input (using a slider)
city_pop = st.slider("City Population", min_value=1000, max_value=1000000, step=1000)

# Convert input data to a DataFrame for prediction
input_data = {
    "merchant": [merchant],
    "category": [category],
    "gender": [gender],
    "city": [city],
    "state": [state],
    "job": [job],
    "amt": [amt],
    "city_pop": [city_pop]  # Include city_pop feature here
}

# Convert dictionary to pandas DataFrame
input_df = pd.DataFrame(input_data)

# Use the loaded model to predict
if st.button("Predict Fraud"):
    # Use the model to make predictions
    prediction = model.predict(input_df)
    st.write(f"Prediction: {'Fraud' if prediction[0] == 1 else 'Not Fraud'}")
    
# Check the input columns
print(input_df.columns)

print(input_df.shape)  # Check the shape of the input data
