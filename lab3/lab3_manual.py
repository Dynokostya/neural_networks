import numpy as np
from sklearn import datasets
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Визначаємо структуру мережі
n_input = 64  # Кількість вхідних ознак для датасету digits
n_hidden = 75
n_output = 10  # Кількість класів для датасету digits

# Завантаження датасету digits
digits = datasets.load_digits()
X_digits = digits.data
Y_digits = digits.target.reshape(-1, 1)

# Стандартизація даних
scaler_digits = StandardScaler().fit(X_digits)
X_digits = scaler_digits.transform(X_digits)

# Розділення даних на навчальний та тестовий набори
X_digits_train, X_digits_test, Y_digits_train, Y_digits_test = train_test_split(X_digits, Y_digits, test_size=0.2, random_state=42)


def to_one_hot(labels, num_classes=None):
    if num_classes is None:
        num_classes = np.max(labels) + 1
    one_hot_array = np.zeros((len(labels), num_classes))
    one_hot_array[np.arange(len(labels)), labels.squeeze()] = 1
    return one_hot_array


# One-hot encoding міток
Y_digits_train_encoded = to_one_hot(Y_digits_train, n_output)
Y_digits_test_encoded = to_one_hot(Y_digits_test, n_output)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    return expZ / expZ.sum(axis=1, keepdims=True)


def initialize_weights(n_input, n_hidden, n_output):
    W1 = np.random.randn(n_input, n_hidden) * 0.01
    b1 = np.zeros((1, n_hidden))
    W2 = np.random.randn(n_hidden, n_output) * 0.01
    b2 = np.zeros((1, n_output))
    return W1, b1, W2, b2


def compute_multiclass_loss(Y, Y_hat):
    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[0]
    L = -(1 / m) * L_sum
    return L


def forward_pass_multiclass(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def backward_pass_multiclass(X, Y, Z1, A1, Z2, A2, W1, W2, b1, b2, learning_rate=0.1):
    m = X.shape[0]
    dZ2 = A2 - Y
    dW2 = 1 / m * np.dot(A1.T, dZ2)
    db2 = 1 / m * np.sum(dZ2, axis=0, keepdims=True)
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(A1)
    dW1 = 1 / m * np.dot(X.T, dZ1)
    db1 = 1 / m * np.sum(dZ1, axis=0, keepdims=True)
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    return W1, b1, W2, b2


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


def predict_multiclass(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_pass_multiclass(X, W1, b1, W2, b2)
    predictions = np.argmax(A2, axis=1)
    return predictions


def confusion_matrix_multiclass(y_true, y_pred):
    K = len(np.unique(y_true))  # Number of classes
    result = np.zeros((K, K))
    for i in range(len(y_true)):
        result[y_true[i], y_pred[i]] += 1
    return result


def precision_recall_f1_multiclass_with_incorrect_classes(y_true, y_pred):
    c_matrix = confusion_matrix_multiclass(y_true, y_pred)
    results = []
    incorrect_classes = []  # Список неправильно класифікованих класів
    for i in range(c_matrix.shape[0]):
        tp = c_matrix[i, i]
        fp = c_matrix[:, i].sum() - tp
        fn = c_matrix[i, :].sum() - tp
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        results.append({
            'class': i,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        # Додавання неправильно класифікованих класів до списку
        if y_true[i] != y_pred[i]:
            incorrect_classes.append(i)

    return results, incorrect_classes


# Для датасету digits
W1, b1, W2, b2, loss_values_digits = train_multiclass_model(X_digits_train, Y_digits_train_encoded, n_input, n_hidden,
                                                            n_output, num_iterations=500, learning_rate=0.1)
Y_digits_pred = predict_multiclass(X_digits_test, W1, b1, W2, b2)

# Визначення точності, відгуку та F1-оцінки
precision, recall, f1, _ = precision_recall_fscore_support(Y_digits_test, Y_digits_pred, average='weighted')

# Виведення метрик
print("---Weighted metrics---")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

results, incorrect_classes = precision_recall_f1_multiclass_with_incorrect_classes(Y_digits_test, Y_digits_pred)

# Виведення результатів
print("---Classes metrics---")
for result in results:
    print(f"Class {result['class']}:")
    print(f"Precision: {result['precision']}")
    print(f"Recall: {result['recall']}")
    print(f"F1 Score: {result['f1']}")

# Виведення неправильно класифікованих класів
print("---Incorrectly Classified Classes---")
for class_idx in incorrect_classes:
    print(f"Class {class_idx} was misclassified.")

plt.plot(loss_values_digits)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss for Digits Dataset')
plt.show()
