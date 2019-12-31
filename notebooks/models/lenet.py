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
"""Implementation of LeNet CNN"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = "Abien Fred Agarap"
__version__ = "1.0.1"

from functools import partial

import tensorflow as tf


conv2d = partial(
        tf.keras.layers.Conv2D,
        activation=tf.nn.relu,
        kernel_initializer="he_normal"
        )


class LeNet(tf.keras.Model):
    def __init__(self, **kwargs):
        super(LeNet, self).__init__()
        self.conv_layer_1 = tf.keras.layers.Conv2D(
            filters=6,
            kernel_size=(5, 5),
            input_shape=(28, 28, 1),
            padding="valid",
            activation=tf.nn.relu,
        )
        self.pool_layer_1 = tf.keras.layers.MaxPooling2D(padding="same")
        self.conv_layer_2 = tf.keras.layers.Conv2D(
            filters=16, kernel_size=(5, 5), padding="valid", activation=tf.nn.relu
        )
        self.pool_layer_2 = tf.keras.layers.MaxPooling2D(padding="same")
        self.flatten = tf.keras.layers.Flatten()
        self.fc_layer_1 = tf.keras.layers.Dense(units=120, activation=tf.nn.relu)
        self.fc_layer_2 = tf.keras.layers.Dense(units=84, activation=tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(
            units=kwargs["num_classes"], activation=tf.nn.softmax
        )

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
