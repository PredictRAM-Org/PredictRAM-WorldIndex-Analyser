# Merge data on Date
merged_data = pd.merge(nsei_data, dji_data, how='inner', on='Date')

# Print column names after merging
st.write("Column names after merging:")
st.write(merged_data.columns)

# Train regression model
X = merged_data.iloc[:, 1:]  # Features (excluding Date and ^NSEI columns)
y_col = 'Open^NSEI'

# Check if the target column exists in the DataFrame
if y_col in merged_data.columns:
    y = merged_data[y_col]

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
else:
    st.error(f"Column '{y_col}' not found in the merged DataFrame.")
