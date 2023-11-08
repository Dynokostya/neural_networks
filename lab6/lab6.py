from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from keras.utils import to_categorical
from keras.models import Sequential
from keras.regularizers import l1, l2
from sklearn.metrics import confusion_matrix, classification_report
from keras.optimizers import Adam, SGD, RMSprop
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Constants for the model
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 256
EPOCHS = 10
VERBOSE = 1

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the data
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Convert class vectors to binary class matrices (one-hot encoding)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Creating the basic neural network model
model = Sequential([
    # Convolutional Layer 1
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
    # Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=l1(0.0001)),
    # Add Dropout or L1/L2 regularization after the Convolutional Layer if needed
    Dropout(0.1),
    BatchNormalization(),

    # Pooling Layer 1
    MaxPooling2D(pool_size=(2, 2)),

    # Convolutional Layer 2
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    # Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l1(0.0001)),
    # Add Dropout or L1/L2 regularization after the Convolutional Layer if needed
    Dropout(0.2),
    BatchNormalization(),


    # Pooling Layer 2
    MaxPooling2D(pool_size=(2, 2)),

    # Convolutional Layer 3
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    # Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l1(0.0001)),
    # Add Dropout or L1/L2 regularization after the Convolutional Layer if needed
    Dropout(0.3),
    BatchNormalization(),

    # Pooling Layer 3
    MaxPooling2D(pool_size=(2, 2)),

    # Flatten the data for the fully-connected layers
    Flatten(),

    # Fully-connected Layer
    Dense(256, activation='relu'),
    # Dense(128, activation='relu', kernel_regularizer=l1(0.0001)),
    # Add Dropout or L1/L2 regularization to the fully-connected layer if needed
    Dropout(0.4),
    BatchNormalization(),

    # Output Layer
    Dense(10, activation='softmax')
])

# Compiling the model
# Using Adam optimizer as default; below are examples of other optimizers
model.compile(optimizer='adam',  # or SGD, RMSprop
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(x_test, y_test),
                    verbose=VERBOSE)

# Quality
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Plot graphs
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.legend()
plt.title('Loss Evolution')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.legend()
plt.title('Accuracy Evolution')
plt.tight_layout()
plt.show()

# Confusion matrix, F1, precision, recall
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

conf_mtx = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mtx, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

print('\nClassification Report:\n')
print(classification_report(y_true, y_pred_classes, digits=4))
