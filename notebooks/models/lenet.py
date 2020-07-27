# Copyright 2019-2020 Abien Fred Agarap
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementation of LeNet CNN"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = "Abien Fred Agarap"
__version__ = "1.0.1"

from functools import partial

import tensorflow as tf


conv2d = partial(
    tf.keras.layers.Conv2D, activation=tf.nn.relu, kernel_initializer="he_normal"
)

dense = partial(tf.keras.layers.Dense, kernel_initializer="he_normal")


class LeNet(tf.keras.Model):
    def __init__(self, **kwargs):
        super(LeNet, self).__init__()
        self.conv_layer_1 = conv2d(
            filters=6, kernel_size=5, input_shape=(28, 28, 1), padding="valid"
        )
        self.pool_layer_1 = tf.keras.layers.MaxPooling2D(padding="same")
        self.conv_layer_2 = conv2d(filters=16, kernel_size=6, padding="valid")
        self.pool_layer_2 = tf.keras.layers.MaxPooling2D(padding="same")
        self.flatten = tf.keras.layers.Flatten()
        self.fc_layer_1 = dense(units=120, activation=tf.nn.relu)
        self.fc_layer_2 = dense(units=84, activation=tf.nn.relu)
        self.output_layer = dense(units=kwargs["num_classes"], activation=tf.nn.softmax)

    def call(self, features):
        activation = self.conv_layer_1(features)
        activation = self.pool_layer_1(activation)
        activation = self.conv_layer_2(activation)
        activation = self.pool_layer_2(activation)
        activation = self.flatten(activation)
        activation = self.fc_layer_1(activation)
        activation = self.fc_layer_2(activation)
        output = self.output_layer(activation)
        return output
