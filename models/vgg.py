from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '1.0.0'

import tensorflow as tf


class VGG16(tf.keras.Model):
    def __init__(self, **kwargs):
        super(VGG16, self).__init__()
        self.conv1_layer_1 = tf.keras.layers.Conv2D(
                64,
                (3, 3),
                input_shape=kwargs['input_shape'],
                padding='same',
                activation=tf.nn.relu
                )
        self.conv1_layer_2 = tf.keras.layers.Conv2D(
                64,
                (3, 3),
                padding='same',
                activation=tf.nn.relu
                )
        self.pool_layer_1 = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='same'
                )
        self.conv2_layer_1 = tf.keras.layers.Conv2D(
                128,
                (3, 3),
                padding='same',
                activation=tf.nn.relu
                )
        self.conv2_layer_2 = tf.keras.layers.Conv2D(
                128,
                (3, 3),
                padding='same',
                activation=tf.nn.relu
                )
        self.pool_layer_2 = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='same'
                )
        self.conv3_layer_1 = tf.keras.layers.Conv2D(
                256,
                (3, 3),
                padding='same',
                activation=tf.nn.relu
                )
        self.conv3_layer_2 = tf.keras.layers.Conv2D(
                256,
                (3, 3),
                padding='same',
                activation=tf.nn.relu
                )
        self.conv3_layer_3 = tf.keras.layers.Conv2D(
                256,
                (3, 3),
                padding='same',
                activation=tf.nn.relu
                )
        self.pool_layer_3 = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='same'
                )
        self.conv4_layer_1 = tf.keras.layers.Conv2D(
                512,
                (3, 3),
                padding='same',
                activation=tf.nn.relu
                )
        self.conv4_layer_2 = tf.keras.layers.Conv2D(
                512,
                (3, 3),
                padding='same',
                activation=tf.nn.relu
                )
        self.conv4_layer_3 = tf.keras.layers.Conv2D(
                512,
                (3, 3),
                padding='same',
                activation=tf.nn.relu
                )
        self.pool_layer_4 = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='same'
                )
        self.conv5_layer_1 = tf.keras.layers.Conv2D(
                512,
                (3, 3),
                padding='same',
                activation=tf.nn.relu
                )
        self.conv5_layer_2 = tf.keras.layers.Conv2D(
                512,
                (3, 3),
                padding='same',
                activation=tf.nn.relu
                )
        self.conv5_layer_3 = tf.keras.layers.Conv2D(
                512,
                (3, 3),
                padding='same',
                activation=tf.nn.relu
                )
        self.pool_layer_5 = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='same'
                )
        self.flatten = tf.keras.layers.Flatten()
        self.fc_layer_1 = tf.keras.layers.Dense(
                units=4096,
                activation=tf.nn.relu
                )
        self.fc_layer_2 = tf.keras.layers.Dense(
                units=4096,
                activation=tf.nn.relu
                )
        self.output_layer = tf.keras.layers.Dense(
                units=kwargs['num_classes'],
                activation=tf.nn.softmax
                )

    @tf.function
    def call(self, features):
        activation = self.conv1_layer_1(features)
        activation = self.conv1_layer_2(activation)
        activation = self.pool_layer_1(activation)
        activation = self.conv2_layer_1(activation)
        activation = self.conv2_layer_2(activation)
        activation = self.pool_layer_2(activation)
        activation = self.conv3_layer_1(activation)
        activation = self.conv3_layer_2(activation)
        activation = self.conv3_layer_3(activation)
        activation = self.pool_layer_3(activation)
        activation = self.conv4_layer_1(activation)
        activation = self.conv4_layer_2(activation)
        activation = self.conv4_layer_3(activation)
        activation = self.pool_layer_4(activation)
        activation = self.conv5_layer_1(activation)
        activation = self.conv5_layer_2(activation)
        activation = self.conv5_layer_3(activation)
        activation = self.pool_layer_5(activation)
        activation = self.flatten(activation)
        activation = self.fc_layer_1(activation)
        activation = self.fc_layer_2(activation)
        output = self.output_layer(activation)
        return output
