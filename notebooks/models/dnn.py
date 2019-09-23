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
"""Implementation of a feed-forward neural network model"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = 'Abien Fred Agarap'
__version__ = '1.0.0'

import tensorflow as tf


class NeuralNet(tf.keras.Model):
    def __init__(self, **kwargs):
        super(NeuralNet, self).__init__()
        self.hidden_layer_1 = tf.keras.layers.Dense(
                units=kwargs['units'][0],
                activation=tf.nn.relu,
                input_shape=kwargs['input_shape']
                )
        self.dropout_layer_1 = tf.keras.layers.Dropout(
                rate=['dropout_rate']
                )
        self.hidden_layer_2 = tf.keras.layers.Dense(
                units=kwargs['units'][1],
                activation=tf.nn.relu
                )
        self.dropout_layer_2 = tf.keras.layers.Dropout(
                rate=kwargs['dropout_rate']
                )
        self.output_layer = tf.keras.layers.Dense(
                units=kwargs['num_classes'],
                activation=tf.nn.softmax
                )

    @tf.function
    def call(self, features):
        activation = self.hidden_layer_1(features)
        activation = self.dropout_layer_1(activation)
        activation = self.hidden_layer_2(activation)
        activation = self.dropout_layer_2(activation)
        output = self.output_layer(activation)
        return output
