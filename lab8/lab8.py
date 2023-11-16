import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Flatten, Reshape, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import fashion_mnist

# Завантаження даних
(x_train, _), (x_test, _) = fashion_mnist.load_data()

# Нормалізація даних
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), 28, 28, 1))
x_test = x_test.reshape((len(x_test), 28, 28, 1))

input_img = Input(shape=(28, 28, 1))

# Архітектура енкодера
x = Conv2D(28, (3, 3), activation='elu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(14, (3, 3), activation='elu', padding='same')(x)
x = Conv2D(7, (3, 3), activation='elu', padding='same')(x)  # Доданий шар
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Архітектура декодера
x = Conv2D(7, (3, 3), activation='elu', padding='same')(encoded)  # Доданий шар
x = Conv2D(14, (3, 3), activation='elu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(28, (3, 3), activation='elu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)


# Модель автоенкодера
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

history = autoencoder.fit(x_train, x_train,
                          epochs=10,
                          batch_size=256,
                          shuffle=True,
                          validation_data=(x_test, x_test))

decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Відображення оригіналу
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Відображення відтворення
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

# Створення рисунка з двома підграфіками
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Перший підграфік - графік функції втрат
ax1.plot(history.history['loss'], label='Train Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_title('Model Loss')
ax1.set_ylabel('Loss')
ax1.set_xlabel('Epochs')
ax1.legend(loc='upper right')

# Другий підграфік - гістограма MSE
mse = np.mean(np.square(x_test - decoded_imgs), axis=(1, 2, 3))
ax2.hist(mse, bins=50)
ax2.set_xlabel('MSE')
ax2.set_ylabel('Кількість зображень')

# Відображення обох підграфіків на одному рисунку
plt.show()

# Розрахунок та відображення MSE між оригінальними та відтвореними зображеннями
mse = np.mean(np.square(x_test - decoded_imgs), axis=(1, 2, 3))
print(f"Середня MSE: {np.mean(mse)}")
print(f"Стандартне відхилення MSE: {np.std(mse)}")
