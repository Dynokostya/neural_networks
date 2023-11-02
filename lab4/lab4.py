from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt

# Завантаження датасету
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Розділення даних на навчальну та тестову вибірки
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Попередня обробка даних: стандартизація ознак
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# Створення моделі нейронної мережі
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    tf.keras.layers.Dense(1)
])

# Експеримент 1: Додавання додаткових шарів
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
#     tf.keras.layers.Dense(30, activation='relu'),
#     tf.keras.layers.Dense(1)
# ])

# Експеримент 2: Зміна кількості нейронів у шарі
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(100, activation='relu', input_shape=X_train.shape[1:]),
#     tf.keras.layers.Dense(1)
# ])

# Експеримент 3: Зміна функції активації
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(30, activation='tanh', input_shape=X_train.shape[1:]),
#     tf.keras.layers.Dense(1)
# ])

# Експеримент 4: Додавання регуляризації
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(30, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01),
#                           input_shape=X_train.shape[1:]),
#     tf.keras.layers.Dense(1)
# ])

# Компіляція моделі
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam())

# Навчання моделі
history = model.fit(X_train_scaled, y_train, epochs=20, validation_data=(X_valid_scaled, y_valid))

# Оцінка якості моделі
mse_test = model.evaluate(X_test_scaled, y_test)
print(f'Test MSE: {mse_test}')

# Прогнозування на тестових даних
y_pred = model.predict(X_test_scaled)
print(f'Predictions: {y_pred.flatten()[:10]}')  # Виведення перших 5 прогнозів

# Візуалізація процесу навчання базової моделі
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='valid')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
