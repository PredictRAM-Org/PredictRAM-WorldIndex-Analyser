# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st
import matplotlib.pyplot as plt
import os

# Function to load data from Excel files in the 'WorldIndex' folder
def load_data(file_name, start_date, end_date):
    file_path = os.path.join('WorldIndex', file_name)
    if os.path.exists(file_path):
        data = pd.read_excel(file_path)
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        data = data.loc[start_date:end_date]
        return data
    else:
        st.error(f"File not found: {file_path}")
        return None

# Streamlit App
st.title('World Index Prediction App')

# User input for date range
start_date_nsei = st.date_input("Select start date for ^NSEI_data.xlsx")
end_date_nsei = st.date_input("Select end date for ^NSEI_data.xlsx", max_value=pd.to_datetime('today'))

start_date_dji = st.date_input("Select start date for ^DJI_data.xlsx")
end_date_dji = st.date_input("Select end date for ^DJI_data.xlsx", max_value=pd.to_datetime('today'))

# Fetch data from Excel files based on user-selected date range
nsei_data = load_data('^NSEI_data.xlsx', start_date_nsei, end_date_nsei)
dji_data = load_data('^DJI_data.xlsx', start_date_dji, end_date_dji)

# Check if data loading was successful
if nsei_data is not None and dji_data is not None:
    # Print unique dates in each DataFrame
    st.write("Unique dates in ^NSEI_data.xlsx:", nsei_data.index.unique())
    st.write("Unique dates in ^DJI_data.xlsx:", dji_data.index.unique())

    # Merge data on Date
    merged_data = pd.merge(nsei_data, dji_data, how='inner', left_index=True, right_index=True)

    # Train regression model
    X = merged_data[['Open_x', 'High_x', 'Low_x', 'Close_x', 'Adj Close_x', 'Volume_x', 
                     'Open_y', 'High_y', 'Low_y', 'Close_y', 'Adj Close_y', 'Volume_y']]

    # Check if the target column exists in the DataFrame
    target_col = 'Open_x'  # Adjust this based on your actual target column
    if target_col in merged_data.columns:
        y = merged_data[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        # Input for predicting opening points
        st.header('Predict Opening Points for ^NSEI')
        st.write('Enter close values for other world indexes:')
        input_data = {}
        for col in X.columns:
            input_data[col] = st.number_input(col, value=0.0)

        # Predict opening points
        if st.button('Predict Opening Points'):
            input_values = [input_data[col] for col in X.columns]
            prediction = model.predict([input_values])[0]
            st.write(f'Predicted Opening Points for ^NSEI: {prediction}')

            # Plotting
            plt.figure(figsize=(10, 6))
            plt.plot(y.index, y.values, label='Actual Opening Prices', marker='o')
            plt.scatter(nsei_data.index[-1], prediction, color='red', label='Predicted Opening Price', marker='x')
            plt.title('Historical and Predicted Opening Prices for ^NSEI')
            plt.xlabel('Date')
            plt.ylabel('Opening Price')
            plt.legend()
            st.pyplot(plt)

    else:
        st.error(f"Column '{target_col}' not found in the merged DataFrame.")
else:
    st.error("Data loading failed.")
