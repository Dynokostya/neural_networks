from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import numpy as np

# Завантаження датасету digits
digits = load_digits()
X = digits.data
Y = digits.target

# Стандартизація даних
scaler = StandardScaler()
X = scaler.fit_transform(X)
max_iter = 1000

# Розділення даних на навчальний та тестовий набори
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Створення та навчання MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(150,), activation='relu', max_iter=max_iter, random_state=42,
                    learning_rate="constant", learning_rate_init=0.01)

losses = []  # Список для зберігання функції втрат на кожній 100-ї ітерації

for i in range(1, max_iter + 1):
    mlp.partial_fit(X_train, Y_train, classes=np.unique(Y_train))
    losses.append(mlp.loss_)
    if i % (max_iter / 10) == 0:
        print(f'Iteration {i}, Loss: {mlp.loss_:.4f}')

# Прогнозування класів для тестового набору
Y_pred = mlp.predict(X_test)

# Визначення точності, відгуку та F1-оцінки
precision, recall, f1, _ = precision_recall_fscore_support(Y_test, Y_pred, average='weighted')

# Виведення метрик
print("---Weighted metrics---")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Метрики для кожного класу
precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(Y_test, Y_pred)

print("---Classes metrics---")
for class_idx, (prec, rec, f1_score) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
    print(f"Class {class_idx}:")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"F1 Score: {f1_score}")

# Візуалізація навчального процесу (функція втрат)
plt.plot(mlp.loss_curve_)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss for Digits Dataset')
plt.show()

# Визначення і виведення неправильно класифікованих класів
incorrectly_classified = Y_test != Y_pred
incorrectly_classified_classes = Y_test[incorrectly_classified]
print("---Incorrectly Classified Classes---")
for class_idx in np.unique(incorrectly_classified_classes):
    print(f"Class {class_idx} was misclassified.")
