from __future__ import division, print_function, absolute_import

import tensorflow as tf
layers = tf.keras.layers
from tensorflow import contrib
autograph = contrib.autograph

import numpy as np
import matplotlib.pyplot as plt

# tf.enable_eager_execution()
# a = tf.random_uniform([20], 0, 1)
# print(a)
# print(a[0:1])
# @autograph.convert()
# def term1_a(self):
#     # a = tf.get_variable('a', [self._num_events], dtype=tf.float32, initializer=tf.zeros_initializer())
#     a = []
#     # We ask you to tell us the element dtype of the list
#     autograph.set_element_type(a, tf.int32)
#
#     for i in tf.range(self._num_events):
#         a[i] = i
#         # a = tf.assign(a[i], tf.dtypes.cast(i, tf.float32))
#
#     # when you're done with the list, stack it
#     # (this is just like np.stack)
#     return autograph.stack(a)
#
#
# print(autograph.to_code(term1_a))


# a = tf.exp(tf.negative(tf.to_float(3)) * tf.to_float(0))

a = tf.random_uniform([5], 0, 1)
# g = tf.reshape(a, [5, 1])
# b = tf.reshape(tf.tile(a, [5]), [5, 5])
# c = tf.matrix_band_part(b[:-1], -1, 0)
# d = tf.reduce_sum(c, axis=1)

b = tf.get_variable('a', tf.shape(a), dtype=tf.float32, initializer=tf.zeros_initializer())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.initialize_local_variables())
    print(sess.run([a, b]))
