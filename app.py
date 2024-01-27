import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import streamlit as st

# Step 1: Fetch Data from WorldIndex using yfinance
nsei_data = yf.download('^NSEI', start='start_date', end='end_date')  # Replace start_date and end_date with your desired date range
world_index_data = yf.download('INDEX_SYMBOL', start='start_date', end='end_date')  # Replace 'INDEX_SYMBOL' and date range

# Step 2: Prepare Data
merged_data = pd.merge(nsei_data, world_index_data, on='Date', how='inner')

# Step 3: Train Regression Model
X = merged_data[['Close_OTHER_INDEX1', 'Close_OTHER_INDEX2', ...]]  # Replace with actual columns
y = merged_data['Open^NSEI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Step 4: Predict Opening Points
# Assuming 'X_predict' contains the close values of other world indexes for prediction
# Replace with actual values
X_predict = merged_data[['Close_OTHER_INDEX1', 'Close_OTHER_INDEX2', ...]]
predicted_open = model.predict(X_predict)

# Step 5: Streamlit App
st.title('World Index Prediction App')

# Display historical data
st.line_chart(merged_data[['Open^NSEI', 'Close_OTHER_INDEX1', 'Close_OTHER_INDEX2']])

# Input for predicting opening points
st.header('Predict Opening Points')
input_data = st.text_input('Enter close values for other world indexes (comma-separated):')
input_values = [float(x) for x in input_data.split(',')]

# Predict opening points
if len(input_values) == len(X.columns):
    prediction = model.predict([input_values])[0]
    st.write(f'Predicted Opening Points: {prediction}')
else:
    st.write('Invalid input. Please enter the correct number of values.')
