import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.utils import to_categorical

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Select an image index for demonstration
image_index = 0

# Show the pixel value distribution before normalization
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.hist(x_train[image_index].ravel(), bins=255, fc='blue', ec='black')
plt.title('Pixel Value Distribution Before Normalization')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

# Normalize the data
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Show the pixel value distribution after normalization
plt.subplot(1, 2, 2)
plt.hist(x_train[image_index].ravel(), bins=255, fc='blue', ec='black')
plt.title('Pixel Value Distribution After Normalization')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
