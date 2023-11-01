import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Завантаження та підготовка даних
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"

# Завантаження даних
data = np.genfromtxt(url, delimiter=',', dtype=str)

# Розділення на ознаки та мітки
X = data[:, :-1]
Y = data[:, -1]

# Кодування категоріальних ознак у числовий формат
label_encoders = [LabelEncoder() for _ in range(X.shape[1])]
for i in range(X.shape[1]):
    X[:, i] = label_encoders[i].fit_transform(X[:, i])

# Перетворення міток у числовий формат
label_encoder_Y = LabelEncoder()
Y = label_encoder_Y.fit_transform(Y)

# Стандартизація даних
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Розділення на навчальний та тестовий набори
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Визначення структури нейромережі
n_input = X_train.shape[1]
n_hidden = 150
n_output = len(np.unique(Y_train))

def to_one_hot(labels, num_classes=None):
    if num_classes is None:
        num_classes = np.max(labels) + 1
    one_hot_array = np.zeros((len(labels), num_classes))
    one_hot_array[np.arange(len(labels)), labels.squeeze()] = 1
    return one_hot_array


# One-hot encoding міток
Y_train_encoded = to_one_hot(Y_train, n_output)
Y_test_encoded = to_one_hot(Y_test, n_output)

# Ініціалізація ваг та зсувів
def initialize_weights(n_input, n_hidden, n_output):
    W1 = np.random.randn(n_input, n_hidden) * 0.01
    b1 = np.zeros((1, n_hidden))
    W2 = np.random.randn(n_hidden, n_output) * 0.01
    b2 = np.zeros((1, n_output))
    return W1, b1, W2, b2

# Функція активації sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Похідна від sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Функція активації softmax
def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    return expZ / expZ.sum(axis=1, keepdims=True)

# Прямий прохід мережі
def forward_pass_multiclass(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# Обернений прохід мережі
def backward_pass_multiclass(X, Y, Z1, A1, Z2, A2, W1, W2, b1, b2, learning_rate=0.1):
    m = X.shape[0]
    dZ2 = A2 - Y
    dW2 = 1 / m * np.dot(A1.T, dZ2)
    db2 = 1 / m * np.sum(dZ2, axis=0, keepdims=True)
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = 1 / m * np.dot(X.T, dZ1)
    db1 = 1 / m * np.sum(dZ1, axis=0, keepdims=True)
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    return W1, b1, W2, b2

# Обчислення функції втрат для багатокласової класифікації
def compute_multiclass_loss(Y, Y_hat):
    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[0]
    L = -(1 / m) * L_sum
    return L

# Тренування мережі
def train_multiclass_model(X, Y, n_input, n_hidden, n_output, num_iterations=100000, learning_rate=0.1):
    W1, b1, W2, b2 = initialize_weights(n_input, n_hidden, n_output)
    losses = []
    for i in range(num_iterations):
        Z1, A1, Z2, A2 = forward_pass_multiclass(X, W1, b1, W2, b2)
        loss = compute_multiclass_loss(Y, A2)
        losses.append(loss)
        W1, b1, W2, b2 = backward_pass_multiclass(X, Y, Z1, A1, Z2, A2, W1, W2, b1, b2, learning_rate)
        if i % 100 == 0:
            print(f'Iteration {i}, Loss: {loss:.4f}')
    return W1, b1, W2, b2, losses

# Прогнозування класів
def predict_multiclass(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_pass_multiclass(X, W1, b1, W2, b2)
    predictions = np.argmax(A2, axis=1)
    return predictions

# Оцінка точності, precision, recall та F1-score
def evaluate_model(Y_true, Y_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    accuracy = accuracy_score(Y_true, Y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(Y_true, Y_pred, average='weighted', zero_division=1)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# Навчання моделі
W1, b1, W2, b2, loss_values_car = train_multiclass_model(X_train, Y_train_encoded, n_input, n_hidden,
                                                         n_output, num_iterations=100, learning_rate=0.001)

# Прогнозування на тестовому наборі
Y_car_pred = predict_multiclass(X_test, W1, b1, W2, b2)

# Оцінка точності моделі
evaluation_result = evaluate_model(Y_test, Y_car_pred)
print("Car Evaluation Database metrics:")
print(evaluation_result)

# Візуалізація функції втрат
plt.plot(loss_values_car)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss for Car Evaluation Database')
plt.show()
