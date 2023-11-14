import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
import matplotlib.pyplot as plt

np.random.seed(0)
# Data Preparation
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
y_train = y_train.squeeze()
y_test = y_test.squeeze()


# Create Base Network
def create_base_model(input_shape):
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, activation='softmax', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='softmax', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='softmax', kernel_regularizer=l2(0.0001))(x)
    return Model(input, x)


base_model = create_base_model((32, 32, 3))


def triplet_loss(y_true, y_pred, alpha=0.4):
    total_length = y_pred.shape.as_list()[-1]
    anchor, positive, negative = (y_pred[:, :int(total_length * 1 / 3)],
                                  y_pred[:, int(total_length * 1 / 3):int(total_length * 2 / 3)],
                                  y_pred[:, int(total_length * 2 / 3):])
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    return loss


# Siamese Network
anchor_input = Input(shape=(32, 32, 3), name='anchor_input')
positive_input = Input(shape=(32, 32, 3), name='positive_input')
negative_input = Input(shape=(32, 32, 3), name='negative_input')

anchor_embedding = base_model(anchor_input)
positive_embedding = base_model(positive_input)
negative_embedding = base_model(negative_input)

merged_vector = tf.keras.layers.concatenate([anchor_embedding, positive_embedding, negative_embedding], axis=-1)
snn = Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_vector)
snn.compile(optimizer=Adam(0.0001), loss=triplet_loss)


def get_triplets(data, labels):
    anchor_images = []
    positive_images = []
    negative_images = []

    unique_labels = np.unique(labels)

    for label in unique_labels:
        same_class_idx = np.where(labels == label)[0]
        diff_class_idx = np.where(labels != label)[0]

        for anchor_idx in same_class_idx:
            positive_idx = np.random.choice(same_class_idx)
            negative_idx = np.random.choice(diff_class_idx)

            anchor_img = data[anchor_idx]
            positive_img = data[positive_idx]
            negative_img = data[negative_idx]

            anchor_images.append(anchor_img)
            positive_images.append(positive_img)
            negative_images.append(negative_img)

    return [np.array(anchor_images), np.array(positive_images), np.array(negative_images)]


triplet_train = get_triplets(x_train, y_train)
triplet_test = get_triplets(x_test, y_test)

# Train model
history = snn.fit(triplet_train, np.zeros(len(triplet_train[0])), epochs=20,
                  batch_size=32, validation_data=(triplet_test, np.zeros(len(triplet_test[0]))))

# Visualization
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

# Accuracy Calculation
# Getting the embeddings of the test data
anchor_embeddings = base_model.predict(triplet_test[0])
positive_embeddings = base_model.predict(triplet_test[1])
negative_embeddings = base_model.predict(triplet_test[2])

# Computing the distances
pos_dist = np.linalg.norm(anchor_embeddings - positive_embeddings, axis=1)
neg_dist = np.linalg.norm(anchor_embeddings - negative_embeddings, axis=1)

# Accuracy computation
correct_predictions = np.sum(pos_dist < neg_dist)
accuracy = correct_predictions / len(pos_dist)
print(f"Accuracy: {accuracy * 100:.2f}%")
