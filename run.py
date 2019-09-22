# Analyzing Deep Learning Models based on Trust Score
# Copyright (C) 2019  Abien Fred Agarap
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"Visualization tool for understanding trust score"
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = 'Abien Fred Agarap'
__version__ = '1.0.0'

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf

from notebooks.models.dnn import NeuralNet
from notebooks.models.lenet import LeNet
from notebooks.models.mini_vgg import MiniVGG
from notebooks.trustscore import TrustScore


tf.random.set_seed(42)
np.random.seed(42)


def load_data():
    (train_features, train_labels),\
            (test_features, test_labels) =\
            tf.keras.datasets.mnist.load_data()
    train_features = train_features.astype('float32') / 255.
    train_labels = tf.keras.utils.to_categorical(train_labels)
    test_features = test_features.astype('float32') / 255.
    test_labels = tf.keras.utils.to_categorical(test_labels)

    pca = PCA(n_components=64)
    enc_train_features = pca.fit_transform(
            train_features.reshape(
                -1, train_features.shape[1] * train_features.shape[2]
                )
            )
    enc_test_features = pca.transform(
            test_features.reshape(
                -1, test_features.shape[1] * test_features.shape[2]
                )
            )
    print('[INFO] Loaded dataset.')
    return (train_features, train_labels),\
           (test_features, test_labels),\
           (enc_train_features, enc_test_features)


def load_model(model_name, model_path, num_classes=10, **kwargs):
    if (model_name == 'LeNet') or (model_name == 'lenet'):
        model = LeNet(num_classes=num_classes)
    elif (model_name == 'MiniVGG') or (model_name == 'mini_vgg'):
        assert 'input_shape' in kwargs,\
                'Expected argument : [input_shape]'
        input_shape = kwargs['input_shape']
        model = MiniVGG(input_shape=input_shape, num_classes=num_classes)
    elif (model_name == 'NeuralNet') or (model_name == 'dnn'):
        assert 'input_shape' in kwargs,\
                'Expected argument : [input_shape]'
        assert 'units' in kwargs,\
            'Expected argument : [units]'
        assert 'dropout_rate' in kwargs,\
            'Expected argument : [dropout_rate]'
        input_shape = kwargs['input_shape']
        units = kwargs['units']
        dropout_rate = kwargs['dropout_rate']
        model = NeuralNet(
                input_shape=input_shape,
                units=units,
                dropout_rate=dropout_rate,
                num_classes=num_classes
                )
    model.load_weights(model_path)
    model.trainable = False
    print('[INFO] Loaded trained {} from {}'.format(
        model_name,
        model_path
        ))
    return model


def fit_ts_model(train_features, train_labels, alpha=5e-2):
    ts = TrustScore(alpha=alpha)
    ts.fit(train_features, train_labels)
    print('[INFO] Fitted trust score model.')
    return ts


def get_prediction(model, test_features, index=None):
    print('[INFO] Getting prediction.')
    return model(test_features)


def get_trust_score(ts_model, test_features, predictions):
    trust_score, closest_not_pred,\
            pred_idx, closest_not_pred_idx,\
            d_to_pred, d_to_closest_not_pred = ts_model.score(
                    test_features.reshape(-1, 64),
                    predictions.numpy().reshape(1, -1)
                    )
    print('[INFO] Computed trust score.')
    return trust_score, closest_not_pred,\
        pred_idx, closest_not_pred_idx,\
        d_to_pred, d_to_closest_not_pred


def visualize_trust_score(
        test_features,
        enc_test_features,
        test_label,
        predictions,
        closest_not_pred,
        trust_score,
        index,
        pred_idx,
        closest_not_pred_idx,
        ):

    print('[INFO] Visualizing prediction and trust score.')
    predictions = predictions.numpy().reshape(-1)

    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(test_features[index].reshape(28, 28), cmap='gray')
    plt.title('label : {}'.format(tf.argmax(test_label[index])))
    plt.subplot(132)
    plt.imshow(test_features[pred_idx].reshape(28, 28), cmap='gray')
    plt.title('predicted : {} ({:.6f})\ntrust score : {:.6f}'.format(
        tf.argmax(predictions).numpy(),
        tf.math.reduce_max(predictions),
        trust_score
    ))
    plt.subplot(133)
    plt.imshow(
            test_features[closest_not_pred_idx].reshape(28, 28), cmap='gray'
            )
    plt.title('closest not predicted : {}'.format(closest_not_pred))
    plt.show()

    enc_test_features = np.array([
        [enc_test_features[index][0], enc_test_features[index][1]],
        [enc_test_features[pred_idx][0], enc_test_features[pred_idx][1]],
        [enc_test_features[closest_not_pred_idx][0],
            enc_test_features[closest_not_pred_idx][1]]
        ])
    labels = ['true_class', 'predicted_class', 'closest_not_predicted']
    figure, axes = plt.subplots()
    axes.scatter(
            enc_test_features[:, 0],
            enc_test_features[:, 1],
            c=np.arange(3)
            )
    for x_i, y_i, label in zip(
            enc_test_features[:, 0],
            enc_test_features[:, 1],
            labels
            ):
        axes.annotate(str(label), xy=(x_i, y_i))
    plt.grid()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
            description='Visualization tool for understanding trust score'
            )
    group = parser.add_argument_group('Parameters')
    group.add_argument(
            '-m',
            '--model',
            required=False,
            default='LeNet',
            type=str,
            help='the model to use, default : [LeNet]')
    group.add_argument(
            '-p',
            '--model_path',
            required=False,
            default='notebooks/saved_model/mnist/lenet/1',
            type=str,
            help='the path to the trained model,'
                    ' default : [notebooks/saved_model/mnist/lenet/1]'
            )
    group.add_argument(
            '-i',
            '--index',
            required=True,
            type=int,
            help='the index of the example to classify'
            )
    arguments = parser.parse_args()
    return arguments


def main(arguments):
    model = arguments.model
    model_path = arguments.model_path
    index = arguments.index

    (train_features, train_labels),\
        (test_features, test_labels),\
        (enc_train_features, enc_test_features) = load_data()

    if (model == 'LeNet') or (model == 'lenet'):
        model = load_model(
                model_name=model,
                model_path=model_path
                )
        prediction = get_prediction(
                model,
                test_features[index].reshape(-1, 28, 28, 1)
                )
    elif (model == 'MiniVGG') or (model == 'mini_vgg'):
        model = load_model(
                model_name=model,
                model_path=model_path,
                input_shape=(28, 28, 1)
                )
        prediction = get_prediction(
                model,
                test_features[index].reshape(-1, 28, 28, 1)
                )
    elif (model == 'NeuralNet') or (model == 'dnn'):
        model = load_model(
                model_name=model,
                model_path=model_path,
                input_shape=(784,),
                units=[512, 512],
                dropout_rate=2e-1
                )
        test_features = test_features.reshape(-1, 784)
        prediction = get_prediction(
                model,
                test_features[index].reshape(-1, 784)
                )

    ts_model = fit_ts_model(
            enc_train_features,
            train_labels
            )

    trust_score, closest_not_pred,\
        pred_idx, closest_not_pred_idx,\
        d_to_pred, d_to_closest_not_pred = get_trust_score(
                ts_model,
                enc_test_features[index],
                prediction
                )
    visualize_trust_score(
            test_features,
            enc_test_features,
            test_labels,
            prediction,
            closest_not_pred[0],
            trust_score[0],
            index,
            pred_idx[0],
            closest_not_pred_idx[0]
            )


if __name__ == '__main__':
    arguments = parse_args()
    main(arguments)
