# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st
import os

# Function to load data from Excel files in the 'WorldIndex' folder
def load_data(file_name):
    file_path = os.path.join('WorldIndex', file_name)
    if os.path.exists(file_path):
        data = pd.read_excel(file_path)
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        return data
    else:
        st.error(f"File not found: {file_path}")
        return None

# Fetch data from Excel files
nsei_data = load_data('^NSEI_data.xlsx')
dji_data = load_data('^DJI_data.xlsx')

# Check if data loading was successful
if nsei_data is not None and dji_data is not None:
    # Merge data on Date
    merged_data = pd.merge(nsei_data, dji_data, how='inner', on='Date')

    # Train regression model
    X = merged_data.iloc[:, 1:]  # Features (excluding Date and ^NSEI columns)
    y = merged_data['Open^NSEI']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Streamlit App
    st.title('World Index Prediction App')

    # Input for predicting opening points
    st.header('Predict Opening Points for ^NSEI')
    st.write('Enter close values for other world indexes:')
    input_data = {}
    for index_col in X.columns:
        input_data[index_col] = st.number_input(index_col, value=0.0)

    # Predict opening points
    if st.button('Predict Opening Points'):
        input_values = [input_data[col] for col in X.columns]
        prediction = model.predict([input_values])[0]
        st.write(f'Predicted Opening Points for ^NSEI: {prediction}')
