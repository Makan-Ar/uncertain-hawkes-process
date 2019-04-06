from __future__ import division, print_function, absolute_import

import tensorflow as tf
layers = tf.keras.layers
from tensorflow import contrib
import tensorflow_probability as tfp
tfd = tfp.distributions

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

# a = tf.random_uniform([5], 0, 1)
# # g = tf.reshape(a, [5, 1])
# # b = tf.reshape(tf.tile(a, [5]), [5, 5])
# # c = tf.matrix_band_part(b[:-1], -1, 0)
# # d = tf.reduce_sum(c, axis=1)
#
# b = tf.get_variable('a', tf.shape(a), dtype=tf.float32, initializer=tf.zeros_initializer())
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     sess.run(tf.initialize_local_variables())
#     print(sess.run([a, b]))


# alpha = [.5, .5]
# dist = tfd.Dirichlet(alpha)
#
# a = dist.sample(1)
# b = dist.prob([.3, .7])
# c = dist.prob([.7, 0.3])
# d = dist.prob(0.3)
# e = dist.prob(.7)
#
# sample_prob_1 = 0.3
# sample_prob_2 = tf.subtract(1., sample_prob_1, name='p2')
# stacked_p_rv = tf.stack([sample_prob_1, sample_prob_2], name='p_stacked')
# g = dist.prob(stacked_p_rv)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     sess.run(tf.initialize_local_variables())
#     print(sess.run([a, b, c, d, e]))


# Simulate two exponential distribution as side information

poisson_exp = np.random.exponential(1. / 6., 5)
hawkes_exp = np.random.exponential(1. / 2., 5)

# mixed exponential
mixed_expo = np.concatenate((poisson_exp, hawkes_exp), axis=0)
mixed_expo = tf.convert_to_tensor(mixed_expo, dtype=tf.float32)
mixed_expo_reshape = tf.reshape(mixed_expo, [tf.size(mixed_expo), 1])

dist = tfd.Exponential(rate=[6., 2.])

# prob_assignment_1 = dist.prob([[5], [1]])
prob_assignment = dist.log_prob(mixed_expo_reshape)
# prob_assignment_2 = dist_2.prob(tf.cast(dataset, dtype='float64'))

# probs_assignments = tf.subtract(tf.cast(1., dtype='float64'), tf.div(prob_assignment_2,
#                                                                      tf.add_n(
#                                                                          [prob_assignment_1, prob_assignment_2])))
a = tf.math.argmax(
    prob_assignment,
    axis=1,
    output_type=tf.dtypes.int32
)
v = tf.boolean_mask(
    mixed_expo,
    a,
    name='boolean_mask',
    axis=None
)
# v = mixed_expo[a]

with tf.Session() as sess:
    print(sess.run([prob_assignment, a, mixed_expo, v]))

print(hawkes_exp)
