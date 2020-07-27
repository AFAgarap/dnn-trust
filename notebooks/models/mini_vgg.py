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
"""Implementation of VGG-like CNN"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = "Abien Fred Agarap"
__version__ = "1.0.1"

import tensorflow as tf


class MiniVGG(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MiniVGG, self).__init__()
        self.conv1_layer_1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation=tf.nn.relu,
            input_shape=kwargs["input_shape"],
        )
        self.conv1_layer_2 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=(3, 3), activation=tf.nn.relu
        )
        self.pool_layer_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2_layer_1 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=(3, 3), activation=tf.nn.relu
        )
        self.conv2_layer_2 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=(3, 3), activation=tf.nn.relu
        )
        self.pool_layer_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout_layer_1 = tf.keras.layers.Dropout(rate=0.25)
        self.flatten = tf.keras.layers.Flatten()
        self.fc_layer = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.dropout_layer_2 = tf.keras.layers.Dropout(rate=0.50)
        self.output_layer = tf.keras.layers.Dense(
            units=kwargs["num_classes"], activation=tf.nn.softmax
        )

    def call(self, features):
        activation = self.conv1_layer_1(features)
        activation = self.conv1_layer_2(activation)
        activation = self.pool_layer_1(activation)
        activation = self.conv2_layer_1(activation)
        activation = self.conv2_layer_2(activation)
        activation = self.pool_layer_2(activation)
        activation = self.dropout_layer_1(activation)
        activation = self.flatten(activation)
        activation = self.fc_layer(activation)
        activation = self.dropout_layer_2(activation)
        output = self.output_layer(activation)
        return output
