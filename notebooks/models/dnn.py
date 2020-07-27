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
"""Implementation of a feed-forward neural network model"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = "Abien Fred Agarap"
__version__ = "1.0.1"

import tensorflow as tf


class NeuralNet(tf.keras.Model):
    def __init__(self, **kwargs):
        super(NeuralNet, self).__init__()
        self.hidden_layer_1 = tf.keras.layers.Dense(
            units=kwargs["units"][0],
            activation=tf.nn.relu,
            input_shape=kwargs["input_shape"],
        )
        self.dropout_layer_1 = tf.keras.layers.Dropout(rate=kwargs["dropout_rate"])
        self.hidden_layer_2 = tf.keras.layers.Dense(
            units=kwargs["units"][1], activation=tf.nn.relu
        )
        self.dropout_layer_2 = tf.keras.layers.Dropout(rate=kwargs["dropout_rate"])
        self.output_layer = tf.keras.layers.Dense(
            units=kwargs["num_classes"], activation=tf.nn.softmax
        )

    def call(self, features):
        activation = self.hidden_layer_1(features)
        activation = self.dropout_layer_1(activation)
        activation = self.hidden_layer_2(activation)
        activation = self.dropout_layer_2(activation)
        output = self.output_layer(activation)
        return output
