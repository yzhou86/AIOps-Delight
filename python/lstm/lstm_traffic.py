import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Dropout, Input
from keras.api.optimizers import Adam
from keras.api.regularizers import l2

# Assuming you have a pandas DataFrame 'df' with the specified columns
# Replace this with your actual data loading code
df = pd.read_csv('dataset.csv')

# Features and target
features = ['passenger_ratio','count_v1','passenger_ratio_v1','L1','count_v2','passenger_ratio_v2','L2','month','day','hour','minute','week','weekday','holiday','freeday','offday','workday']
target = 'count'

# Normalize the features
feature_scaler = MinMaxScaler()
df[features] = feature_scaler.fit_transform(df[features])

target_scaler = MinMaxScaler()
df[[target]] = target_scaler.fit_transform(df[[target]])


def create_sequences(data, target, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = target[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10  # Number of previous time steps to use for prediction
X, y = create_sequences(df[features].values, df[target].values, seq_length)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build the LSTM model
model = Sequential()
model.add(Input(shape=(X.shape[1], X.shape[2])))  # Input shape (seq_length, number of features)
model.add(LSTM(units=50, activation='relu', return_sequences=True, kernel_regularizer=l2(0.001)))
model.add(LSTM(units=50, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error', optimizer=optimizer)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)

# Predict on validation data
y_val_pred = model.predict(X_val)

# Inverse transform the predictions and actual values to get them back to the original scale
y_val_pred = target_scaler.inverse_transform(y_val_pred)
y_val_actual = target_scaler.inverse_transform(y_val.reshape(-1, 1))

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_val_actual, y_val_pred))
print(f"Validation RMSE: {rmse}")

# Example of making a prediction with real-time data
real_time_data = np.random.rand(seq_length, len(features))  # Example: Random data, replace with actual
real_time_data = feature_scaler.transform(real_time_data)
real_time_data = real_time_data.reshape((1, seq_length, len(features)))

prediction = model.predict(real_time_data)

# Inverse transform the prediction to get the actual traffic_count value
predicted_traffic_count = target_scaler.inverse_transform(prediction)
print("Predicted count:", predicted_traffic_count[0][0])
