from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '1.0.0'

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)

data = loadmat('/Users/darth/Projects/matlab/emnist-letters.mat')

train_features = data['dataset'][0][0][0][0][0][0]
test_features = data['dataset'][0][0][1][0][0][0]

train_labels = data['dataset'][0][0][0][0][0][1]
test_labels = data['dataset'][0][0][1][0][0][1]

train_features = train_features.reshape(-1, 28, 28, 1)
test_features = test_features.reshape(-1, 28, 28, 1)

train_features = train_features / 255.
test_features = test_features / 255.

train_labels = train_labels - 1
test_labels = test_labels - 1

validation_features, test_features, validation_labels, test_labels = \
        train_test_split(test_features, test_labels, test_size=0.50, stratify=test_labels, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(6, (5, 5), input_shape=(28, 28, 1), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(16, (5, 5), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=120, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=84, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=26, activation=tf.nn.softmax)
    ])
model.load_weights('notebooks/saved_model/emnist-letters/lenet/1')

model.compile(loss=tf.losses.sparse_categorical_crossentropy,
              optimizer=tf.optimizers.SGD(learning_rate=1e-1, momentum=9e-1, decay=1e-6),
              metrics=['sparse_categorical_accuracy'])
# model.fit(train_features,
#           train_labels,
#           batch_size=512,
#           epochs=50,
#           validation_data=(validation_features, validation_labels))
score = model.evaluate(test_features, test_labels, batch_size=32)
print('accuracy : {}'.format(score[1]))
