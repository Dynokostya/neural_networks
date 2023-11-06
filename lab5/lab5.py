from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.initializers import HeNormal, GlorotUniform
from keras.regularizers import l1, l2
import matplotlib.pyplot as plt
import pandas as pd

# Constants for the model
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 100
VERBOSE = 1

# Fetching the California housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Preprocessing the data: removing missing values (if any)
X_filled = pd.DataFrame(X).ffill().values
y_filled = pd.DataFrame(y).ffill().values

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_filled, y_filled, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# Normalizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creating the basic neural network model
model = Sequential([
    # Xavier initialization
    # Dense(64, input_shape=(X_train_scaled.shape[1],), activation='relu', kernel_initializer=GlorotUniform()),
    # Input layer with He initialization
    Dense(64, input_shape=(X_train_scaled.shape[1],), activation='relu', kernel_initializer=HeNormal()),

    BatchNormalization(),
    Dropout(0.5),

    # Hidden layers with L1 regularization
    # Dense(64, activation='relu', kernel_regularizer=l1(0.01)),
    # Hidden layers with L2 regularization
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),

    BatchNormalization(),
    Dropout(0.3),

    # Output layer
    Dense(1)
])

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    verbose=VERBOSE)

# Evaluating the model on test data
test_loss = model.evaluate(X_test_scaled, y_test, verbose=0)
y_pred = model.predict(X_test_scaled).flatten()
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Plotting training and validation loss over epochs
plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Output evaluation metrics
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Test Loss:", test_loss)
