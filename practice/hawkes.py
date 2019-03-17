from __future__ import division, print_function, absolute_import

"""The Hawkes process class."""

import contextlib
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import dtype_util

# from tensorflow import contrib
# autograph = contrib.autograph


class Hawkes():
    def __init__(self,
                 event_times,
                 background_intensity,
                 alpha,
                 beta,
                 name="Hawkes"):

        with tf.name_scope(name, values=[event_times, background_intensity, alpha, beta]) as name:
            _dtype = dtype_util.common_dtype([event_times, background_intensity, alpha, beta], tf.float32)
            self._event_times = tf.convert_to_tensor(event_times, name="event_times", dtype=_dtype)
            self._bg_intensity = tf.convert_to_tensor(background_intensity, name="background_intensity", dtype=_dtype)
            self._alpha = tf.convert_to_tensor(alpha, name="alpha", dtype=_dtype)
            self._beta = tf.convert_to_tensor(beta, name="beta", dtype=_dtype)

        self._num_events = len(event_times)
        self._dtype = _dtype
        self._graph_parents = [self._event_times, self._bg_intensity, self._alpha, self._beta]
        self._name = name

    @property
    def background_intensity(self):
        return self._event_times

    @property
    def background_intensity(self):
        return self._bg_intensity

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    def cum_log_likelihood(self, name='cum_log_likelihood'):
        # based on https://arxiv.org/abs/1507.02822 eq 21
        # full negative log likelihood is term 1 - term 2 + term 3
        # term 1: sum (i) from 0 to t(log(bg_intensity + alpha * sum (j) from 0 to (exp(-beta * (ti - tj))))
        # term 2: bg_intensity * t
        # term 3: (alpha / beta) * (-ind(t) + sum (i) from 0 to t (exp(-beta * (t - ti))))
        with self._name_scope(name):
            term1 = self.evaluate_first_term()
            term2 = self.evaluate_second_term()
            term3 = self.evaluate_third_term()

            nagtive_log_likelihood = term1 - term2 + term3

            return term1, term2, term3, nagtive_log_likelihood

    def evaluate_first_term(self, name="evaluate_first_term"):
        with tf.variable_scope("hawkes_ll_first_term"):
            a = tf.get_variable('a', [self._num_events], dtype=tf.float32, initializer=tf.zeros_initializer())

        with self._name_scope(name, values=[a]):
            def cond(i, iters):
                return tf.less(i, iters)

            def body(i, iters):
                with tf.variable_scope("hawkes_ll_first_term", reuse=tf.AUTO_REUSE):
                    a = tf.get_variable('a')
                a = tf.assign(a[i], tf.math.exp(tf.math.negative(self._beta) *
                                                (self._event_times[i] - self._event_times[i - 1])) * (1. + a[i - 1]))

                with tf.control_dependencies([a]):
                    return [tf.add(i, 1), iters]

            i = tf.constant(1, dtype=tf.int32)
            iters = tf.constant(self._num_events)
            i, _ = tf.while_loop(cond, body, [i, iters], name="compute_A", parallel_iterations=1)

            with tf.variable_scope("hawkes_ll_first_term", reuse=tf.AUTO_REUSE):
                a = tf.get_variable('a')

            with tf.control_dependencies([i]):
                first_term = tf.reduce_sum(tf.log(tf.add(self._bg_intensity, tf.multiply(self._alpha, a))))
            return first_term

    def evaluate_second_term(self, name="evaluate_second_term"):
        with self._name_scope(name):
            return tf.multiply(self._bg_intensity, tf.reduce_sum(self._event_times))

    def evaluate_third_term(self, name="evaluate_third_term"):
        with self._name_scope(name):
            def cond(kernel, i, iters):
                return tf.less(i, iters)

            def body(kernel, i, iters):
                kernel = tf.add(kernel, tf.reduce_sum(tf.exp(tf.negative(self._beta) *
                                                             (self._event_times[i] - self._event_times[0:i]))))
                return [kernel, tf.add(i, 1), iters]

            kernel = tf.constant(0., dtype=tf.float32)
            i = tf.constant(1, dtype=tf.int32)
            num_events = tf.constant(self._num_events)
            kernel, i, _ = tf.while_loop(cond, body, [kernel, i, num_events], name="compute_kernel",
                                         parallel_iterations=1)

            kernel = tf.subtract(kernel, tf.to_float(tf.divide(num_events * (num_events - 1), 2)))
            third_term = tf.multiply(tf.truediv(self._alpha, self._beta), kernel)

            return third_term

    # Taken from tensorflow.probability distribution.py
    @contextlib.contextmanager
    def _name_scope(self, name=None, values=None):
        """Helper function to standardize op scope."""
        with tf.name_scope(self._name):
            with tf.name_scope(name, values=(
                    ([] if values is None else values) + self._graph_parents)) as scope:
                yield scope
