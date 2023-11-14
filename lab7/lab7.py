from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, BatchNormalization
from keras.optimizers import RMSprop
from keras.regularizers import l1, l2
from keras import backend as K
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)
# Data Preparation
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
y_train = y_train.squeeze()
y_test = y_test.squeeze()


# Create Pairs
def create_pairs(x, digit_indices):
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = np.random.randint(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


digit_indices = [np.where(y_train == i)[0] for i in range(10)]
tr_pairs, tr_y = create_pairs(x_train, digit_indices)
digit_indices = [np.where(y_test == i)[0] for i in range(10)]
te_pairs, te_y = create_pairs(x_test, digit_indices)


# Create Base Network
def create_base_network(input_shape):
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


base_network = create_base_network((32, 32, 3))

# Siamese Network
input_a = Input(shape=(32, 32, 3))
input_b = Input(shape=(32, 32, 3))

processed_a = base_network(input_a)
processed_b = base_network(input_b)


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


distance = Lambda(euclidean_distance)([processed_a, processed_b])
model = Model([input_a, input_b], distance)


# Compile and Train the Model
def contrastive_loss(y_true, y_pred):
    margin = 1
    y_true = K.cast(y_true, 'float32')
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


model.compile(loss=contrastive_loss, optimizer=RMSprop())
history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, epochs=20,
                    batch_size=128, validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

# Predictions and Evaluation
predicted = model.predict([te_pairs[:, 0], te_pairs[:, 1]])

# Visualization and Accuracy Calculation
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

threshold = 0.5
binary_predictions = [1 if dist < threshold else 0 for dist in predicted]

accuracy = accuracy_score(te_y, binary_predictions)
precision = precision_score(te_y, binary_predictions)
recall = recall_score(te_y, binary_predictions)
f1 = f1_score(te_y, binary_predictions)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
