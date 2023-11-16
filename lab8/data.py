import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

# Завантаження даних
(x_train, _), (x_test, _) = fashion_mnist.load_data()

# Відобразимо кілька зображень до нормалізації
plt.figure(figsize=(15, 10))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i].reshape(28, 28), cmap='gray', vmin=0, vmax=255)
    plt.title(f"До нормалізації\nМакс: {x_train[i].max():.2f}\nМін: {x_train[i].min():.2f}")
    plt.colorbar()
    plt.axis('off')

# Нормалізація даних
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), 28, 28, 1))
x_test = x_test.reshape((len(x_test), 28, 28, 1))

# Відобразимо кілька зображень після нормалізації
for i in range(5):
    plt.subplot(2, 5, i + 6)
    plt.imshow(x_train[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
    plt.title(f"Після нормалізації\nМакс: {x_train[i].max():.2f}\nМін: {x_train[i].min():.2f}")
    plt.colorbar()
    plt.axis('off')

plt.suptitle('Зображення до та після нормалізації')
plt.show()
