from __future__ import division, print_function, absolute_import

"""The Hawkes process class."""

import contextlib
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import dtype_util


class Hawkes():
    def __init__(self,
                 event_times,
                 background_intensity,
                 alpha,
                 beta,
                 dtype=None,
                 name="Hawkes"):

        with tf.name_scope(name, values=[event_times, background_intensity, alpha, beta]) as name:
            if not dtype:
                _dtype = dtype_util.common_dtype([event_times, background_intensity, alpha, beta], tf.float32)
            else:
                _dtype = dtype
            self._event_times = tf.convert_to_tensor(event_times, name="event_times", dtype=_dtype)
            self._bg_intensity = tf.convert_to_tensor(background_intensity, name="background_intensity", dtype=_dtype)
            self._alpha = tf.convert_to_tensor(alpha, name="alpha", dtype=_dtype)
            self._beta = tf.convert_to_tensor(beta, name="beta", dtype=_dtype)

        self._num_events = tf.size(self._event_times)
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

    def log_likelihood(self, name='log_likelihood'):
        # based on https://arxiv.org/abs/1507.02822 eq 21
        # full negative log likelihood is term 1 - term 2 + term 3
        # term 1: sum (i) from 0 to t(log(bg_intensity + alpha * sum (j) from 0 to (exp(-beta * (ti - tj))))
        # term 2: bg_intensity * t
        # term 3: (alpha / beta) * (-ind(t) + sum (i) from 0 to t (exp(-beta * (t - ti))))
        with self._name_scope(name):
            term1 = self.evaluate_first_term()
            term2 = self.evaluate_second_term()
            term3 = self.evaluate_third_term()

            log_likelihood = term1 - term2 + term3

            return log_likelihood
            # return term1, term2, term3, log_likelihood

    def evaluate_first_term(self, name="evaluate_first_term"):
        with self._name_scope(name):
            def cond(first_term, prev_a_term, i, iters):
                return tf.less(i, iters)

            def body(first_term, prev_a_term, i, iters):
                prev_a_term = tf.exp(tf.negative(self._beta) *
                                     (self._event_times[i] - self._event_times[i - 1])) * (1. + prev_a_term)

                first_term = tf.add(first_term, tf.log(tf.add(self._bg_intensity,
                                                              tf.multiply(self._alpha, prev_a_term))))
                return [first_term, prev_a_term, tf.add(i, 1), iters]

            first_term = tf.constant(0., dtype=self._dtype)
            prev_a_term = tf.constant(0., dtype=self._dtype)
            i = tf.constant(1, dtype=tf.int32)
            # iters = tf.constant(self._num_events)
            first_term, prev_a_term, i, _ = tf.while_loop(cond, body, [first_term, prev_a_term, i, self._num_events],
                                                          name="compute_first_term", parallel_iterations=1)

            # Adding the k = 0 (based on 0 indexing) to the total
            first_term = tf.add(first_term, tf.log(self._bg_intensity))

            return first_term

    # Older implementation of term 1 calculation
    def evaluate_first_term_with_a_term_tensor(self, name="evaluate_first_term_with_a_term_tensor"):
        with tf.variable_scope("hawkes_ll_first_term"):
            # a = tf.get_variable('a', [self._num_events], dtype=self._dtype, initializer=tf.zeros_initializer())
            a = tf.get_variable('a', tf.shape(self._event_times), dtype=self._dtype, initializer=tf.zeros_initializer())

        with self._name_scope(name, values=[a]):
            def cond(i, iters):
                return tf.less(i, iters)

            def body(i, iters):
                with tf.variable_scope("hawkes_ll_first_term", reuse=tf.AUTO_REUSE):
                    a = tf.get_variable('a', dtype=self._dtype)
                a = tf.assign(a[i], tf.math.exp(tf.math.negative(self._beta) *
                                                (self._event_times[i] - self._event_times[i - 1])) * (1. + a[i - 1]))

                with tf.control_dependencies([a]):
                    return [tf.add(i, 1), iters]

            i = tf.constant(1, dtype=tf.int32)
            i, _ = tf.while_loop(cond, body, [i, self._num_events], name="compute_A", parallel_iterations=1)

            with tf.variable_scope("hawkes_ll_first_term", reuse=tf.AUTO_REUSE):
                a = tf.get_variable('a', dtype=self._dtype)

            with tf.control_dependencies([i]):
                first_term = tf.reduce_sum(tf.log(tf.add(self._bg_intensity, tf.multiply(self._alpha, a))))
            return first_term

    def evaluate_first_term_no_loop(self, name="evaluate_first_term_no_loop"):
        with self._name_scope(name):

            a_term_exp = tf.exp(tf.negative(self._beta) * (self._event_times[1:] - self._event_times[:-1]))

            # exp ^ (beta * t) of t_0 to t_k-1
            arrivals_repeat = tf.reshape(tf.tile(self._event_times[:-1], [self._num_events - 1]),
                                         [self._num_events - 1, self._num_events - 1])
            interarrivals_repeat = tf.subtract(tf.reshape(self._event_times[:-1], [self._num_events - 1, 1]),
                                               arrivals_repeat)
            e_beta_repeat = tf.exp(tf.multiply(tf.negative(self._beta), interarrivals_repeat))
            e_beta_repeat_lower_tri = tf.matrix_band_part(e_beta_repeat, -1, 0)
            a_term_sum = tf.reduce_sum(e_beta_repeat_lower_tri, axis=1)
            a_term = tf.multiply(a_term_exp, a_term_sum, name='a_term')

            first_term = tf.reduce_sum(tf.log(tf.add(self._bg_intensity, tf.multiply(self._alpha, a_term))))
            # Adding the k = 0 (based on 0 indexing) to the total
            first_term = tf.add(first_term, tf.log(self._bg_intensity), name='first_term')

            return first_term

    def evaluate_second_term(self, name="evaluate_second_term"):
        with self._name_scope(name):
            return tf.multiply(self._bg_intensity, self._event_times[-1])

    def evaluate_third_term(self, name="evaluate_third_term"):
        with self._name_scope(name):
            kernel = tf.subtract(
                tf.reduce_sum(tf.exp(tf.negative(self._beta) * (self._event_times[-1] - self._event_times))),
                tf.cast(self._num_events, dtype=self._dtype))
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

    # def cum_log_likelihood(self, name='cum_log_likelihood'):
    #     # based on https://arxiv.org/abs/1507.02822 eq 21
    #     # full negative log likelihood is term 1 - term 2 + term 3
    #     # term 1: sum (i) from 0 to t(log(bg_intensity + alpha * sum (j) from 0 to (exp(-beta * (ti - tj))))
    #     # term 2: bg_intensity * t
    #     # term 3: (alpha / beta) * (-ind(t) + sum (i) from 0 to t (exp(-beta * (t - ti))))
    #     with self._name_scope(name):
    #         term1 = self.evaluate_first_term()
    #         term2 = self.evaluate_cum_second_term()
    #         term3 = self.evaluate_cum_third_term()
    #
    #         log_likelihood = term1 - term2 + term3
    #
    #         return nagtive_log_likelihood
    #         # return term1, term2, term3, log_likelihood
    #
    # def evaluate_cum_second_term(self, name="evaluate_cum_second_term"):
    #     with self._name_scope(name):
    #         return tf.multiply(self._bg_intensity, tf.reduce_sum(self._event_times))
    #
    # def evaluate_cum_third_term(self, name="evaluate_cum_third_term"):
    #     with self._name_scope(name):
    #         def cond(kernel, i, iters):
    #             return tf.less(i, iters)
    #
    #         def body(kernel, i, iters):
    #             kernel = tf.add(kernel, tf.reduce_sum(tf.exp(tf.negative(self._beta) *
    #                                                          (self._event_times[i] - self._event_times[0:i]))))
    #             return [kernel, tf.add(i, 1), iters]
    #
    #         kernel = tf.constant(0., dtype=self._dtype)
    #         i = tf.constant(1, dtype=tf.int32)
    #         # num_events = tf.constant(self._num_events)
    #         kernel, i, _ = tf.while_loop(cond, body, [kernel, i, self._num_events], name="compute_kernel",
    #                                      parallel_iterations=1)
    #
    #         kernel = tf.subtract(kernel, tf.cast(tf.divide(self._num_events * (self._num_events - 1), 2),
    #                                              dtype=self._dtype))
    #         third_term = tf.multiply(tf.truediv(self._alpha, self._beta), kernel)
    #
    #         return third_term
