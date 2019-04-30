# from __future__ import division, print_function, absolute_import
#
# import tensorflow as tf
# layers = tf.keras.layers
# from tensorflow import contrib
# import tensorflow_probability as tfp
# tfd = tfp.distributions
#
# autograph = contrib.autograph
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# # tf.enable_eager_execution()
# # a = tf.random_uniform([20], 0, 1)
# # print(a)
# # print(a[0:1])
# # @autograph.convert()
# # def term1_a(self):
# #     # a = tf.get_variable('a', [self._num_events], dtype=tf.float32, initializer=tf.zeros_initializer())
# #     a = []
# #     # We ask you to tell us the element dtype of the list
# #     autograph.set_element_type(a, tf.int32)
# #
# #     for i in tf.range(self._num_events):
# #         a[i] = i
# #         # a = tf.assign(a[i], tf.dtypes.cast(i, tf.float32))
# #
# #     # when you're done with the list, stack it
# #     # (this is just like np.stack)
# #     return autograph.stack(a)
# #
# #
# # print(autograph.to_code(term1_a))
#
#
# # a = tf.exp(tf.negative(tf.to_float(3)) * tf.to_float(0))
#
# # a = tf.random_uniform([5], 0, 1)
# # # g = tf.reshape(a, [5, 1])
# # # b = tf.reshape(tf.tile(a, [5]), [5, 5])
# # # c = tf.matrix_band_part(b[:-1], -1, 0)
# # # d = tf.reduce_sum(c, axis=1)
# #
# # b = tf.get_variable('a', tf.shape(a), dtype=tf.float32, initializer=tf.zeros_initializer())
# #
# # with tf.Session() as sess:
# #     sess.run(tf.global_variables_initializer())
# #     sess.run(tf.initialize_local_variables())
# #     print(sess.run([a, b]))
#
#
# # alpha = [.5, .5]
# # dist = tfd.Dirichlet(alpha)
# #
# # a = dist.sample(1)
# # b = dist.prob([.3, .7])
# # c = dist.prob([.7, 0.3])
# # d = dist.prob(0.3)
# # e = dist.prob(.7)
# #
# # sample_prob_1 = 0.3
# # sample_prob_2 = tf.subtract(1., sample_prob_1, name='p2')
# # stacked_p_rv = tf.stack([sample_prob_1, sample_prob_2], name='p_stacked')
# # g = dist.prob(stacked_p_rv)
# #
# # with tf.Session() as sess:
# #     sess.run(tf.global_variables_initializer())
# #     sess.run(tf.initialize_local_variables())
# #     print(sess.run([a, b, c, d, e]))
#
#
# # Simulate two exponential distribution as side information
#
# # poisson_exp = np.random.exponential(1. / 6., 5)
# # hawkes_exp = np.random.exponential(1. / 2., 5)
# #
# # # mixed exponential
# # mixed_expo = np.concatenate((poisson_exp, hawkes_exp), axis=0)
# # mixed_expo = tf.convert_to_tensor(mixed_expo, dtype=tf.float32)
# # mixed_expo_reshape = tf.reshape(mixed_expo, [tf.size(mixed_expo), 1])
# #
# # dist = tfd.Exponential(rate=[6., 2.])
# #
# # # prob_assignment_1 = dist.prob([[5], [1]])
# # prob_assignment = dist.log_prob(mixed_expo_reshape)
# # # prob_assignment_2 = dist_2.prob(tf.cast(dataset, dtype='float64'))
# #
# # # probs_assignments = tf.subtract(tf.cast(1., dtype='float64'), tf.div(prob_assignment_2,
# # #                                                                      tf.add_n(
# # #                                                                          [prob_assignment_1, prob_assignment_2])))
# # a = tf.math.argmax(
# #     prob_assignment,
# #     axis=1,
# #     output_type=tf.dtypes.int32
# # )
# # v = tf.boolean_mask(
# #     mixed_expo,
# #     a,
# #     name='boolean_mask',
# #     axis=None
# # )
# # # v = mixed_expo[a]
# #
# # with tf.Session() as sess:
# #     print(sess.run([prob_assignment, a, mixed_expo, v]))
# #
# # print(hawkes_exp)












# import numpy as np
# from sklearn.impute import SimpleImputer
# from sklearn.ensemble import RandomForestClassifier
#
# # dataset = np.loadtxt('/Users/makanarastuie/Downloads/hepatitis.data')
# dataset = np.genfromtxt('/Users/makanarastuie/Downloads/hepatitis.data',
#                         dtype=float,
#                         invalid_raise=False,
#                         missing_values='?',
#                         usemask=False,
#                         filling_values=np.nan,
#                         delimiter=',')
# print(dataset)
# imp = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
# dataset = imp.fit_transform(dataset)
# print(dataset)
# # clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
# # clf.fit(X, y)
# # RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
# #             max_depth=2, max_features='auto', max_leaf_nodes=None,
# #             min_impurity_decrease=0.0, min_impurity_split=None,
# #             min_samples_leaf=1, min_samples_split=2,
# #             min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
# #             oob_score=False, random_state=0, verbose=0, warm_start=False)
# # >>> print(clf.feature_importances_)
# # [0.14205973 0.76664038 0.0282433  0.06305659]
# # >>> print(clf.predict([[0, 0, 0, 0]]))

# from tick.hawkes import SimuHawkesExpKernels, SimuPoissonProcess
#
# time = 100
# num_noise = 40
#
# lambd = num_noise / 100
# poisson = SimuPoissonProcess(lambd, verbose=False, end_time=time, seed=None)
# poisson.simulate()
#
# print(poisson.timestamps[0])
# print(len(poisson.timestamps[0]))
# print(len(poisson.timestamps[0]) / time)



import hawkes as hwk
from sklearn.metrics import f1_score
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions

import sys
sys.path.insert(0, r'/nethome/marastu2/uncertain-hawkes-process')
from hawkes_uncertain_simulator import HawkesUncertainModel

plot_base_path = '/shared/Results/HawkesUncertainEvents/temp'

_h_intensity = 1
_h_beta = 2
_h_alpha = 0.9

_runtime = 10

_p_intensity = 0.3

_h_exp_rate = 1.5
_p_exp_rate = 3.5

hum = HawkesUncertainModel(h_lambda=_h_intensity, h_alpha=_h_alpha, h_beta=_h_beta, h_exp_rate=_h_exp_rate,
                           p_lambda=_p_intensity, p_exp_rate=_p_exp_rate,
                           noise_percentage_ub=0.2, run_time=_runtime, delta=0.01)

# event_times = tf.convert_to_tensor(hum.hawkes.timestamps[0], name="event_times_data", dtype=tf.float32)
# events_side_info = tf.convert_to_tensor(hum.hawkes_exp, name="event_side_data", dtype=tf.float32)

event_times = tf.convert_to_tensor(hum.mixed_timestamps, name="event_times_data", dtype=tf.float32)
events_side_info = tf.convert_to_tensor(hum.mixed_expo, name="event_side_data", dtype=tf.float32)
print("Noise Percentage: ", hum.noise_percentage)

# with tf.Session() as sess:
#     print(sess.run([tf.shape(events_side_info)[0], tf.shape(tf.transpose(events_side_info))]))
#
# exit()

# print(hum.hawkes.timestamps[0])
# exit()
# rv_hawkes_observations = hwk.Hawkes(_h_intensity,
#                                     _h_alpha,
#                                     _h_beta,
#                                     tf.float32, name="hawkes_observations_rv")
#
# g = rv_hawkes_observations.log_likelihood(tf.convert_to_tensor([1.88874331]))
#
# with tf.Session() as sess:
#     print(sess.run(g))
# exit()
# cat_prob = 1. - hum.noise_percentage
# # cat_prob = 0.5
# print(cat_prob)
# stacked_p_rv = tf.stack([cat_prob, 1. - cat_prob], name='p_stacked')
#
# multi = tfd.Multinomial(total_count=1., probs=stacked_p_rv)
# categ = tfd.Categorical(probs=stacked_p_rv)
#

# tensor = [1.2, 2, 3, 4]
# # mask = np.array([1, 0, 0, 1])
# # g = tf.boolean_mask(tensor, mask)  # [[1, 2], [5, 6]]
# po = tf.convert_to_tensor(tensor, tf.float32)
# h = tf.concat([po, [4.2]], axis=0)
# e = tf.argmax([0.5, 0.9])
# kj = tf.equal(e, 0)
# with tf.Session() as sess:
#     print(sess.run([e, kj]))
# #
# # print(cat_prob, hum.noise_percentage)
# exit()


def sample_z(events_t, events_info,
             hw_sample_alpha, hw_sample_beta, hw_sample_intensity,
             poi_sample_intensity,
             exp_hw_sample_rate, exp_poi_sample_rate,
             hawkes_cat_prob):

    num_events = events_t.get_shape()[0]

    # The order of likelihoods are switched here to accommodate the hawkes mask
    # compute log_prob of cat assignment, due to the exp mixture model
    exp_dists = tfd.Exponential(rate=[exp_poi_sample_rate, exp_hw_sample_rate])
    exp_log_prob = exp_dists.log_prob(tf.reshape(events_info, [num_events, 1]))
    stacked_cat_probs = tf.log(tf.stack([1. - hawkes_cat_prob, hawkes_cat_prob], name='p_stacked'))

    exp_mixture_log_prob = stacked_cat_probs + exp_log_prob

    rv_hawkes_observations = hwk.Hawkes(hw_sample_intensity,
                                        hw_sample_alpha,
                                        hw_sample_beta,
                                        tf.float32, name="hawkes_observations_rv")

    rv_exp_poisson_inter_arrival = tfd.Exponential(poi_sample_intensity)

    def cond(i, iters):
        return tf.less(i, iters)

    def body(i, iters):
        with tf.variable_scope("sample_latent_variable", reuse=tf.AUTO_REUSE):
            hawkes_mask = tf.get_variable('hawkes_mask', num_events, dtype=tf.int32,
                                          initializer=tf.zeros_initializer())
            prev_hawkes_ll = tf.get_variable('prev_hawkes_ll', [], dtype=tf.float32, initializer=tf.zeros_initializer())

            last_noise_timestamp = tf.get_variable('last_noise_timestamp', [], dtype=tf.float32,
                                                   initializer=tf.zeros_initializer())

            mixture_log_prob = tf.get_variable('mixture_log_prob', [num_events, 2], dtype=tf.float32,
                                               initializer=tf.zeros_initializer())

        # Hawkes log-likelihood
        hawkes_history = tf.boolean_mask(events_t, hawkes_mask)
        hawkes_ll = rv_hawkes_observations.log_likelihood(tf.concat([hawkes_history, [events_t[i]]], axis=0))

        # Poisson log-likelihood
        poisson_ll = rv_exp_poisson_inter_arrival.log_cdf(events_t[i] - last_noise_timestamp)

        # TODO: check whether to use the entire Hawkes ll, or just the difference with the last one
        event_assignment_log_prob = tf.stack([poisson_ll, hawkes_ll - prev_hawkes_ll], name='point_process_ll')
        event_assignment_log_prob = event_assignment_log_prob + exp_mixture_log_prob[i]
        # event_assignment_log_prob = event_assignment_log_prob

        # for logging purposes
        mixture_log_prob = tf.assign(mixture_log_prob[i], event_assignment_log_prob)

        # 0 for poisson/noise, 1 for hawkes (order is changed here to accommodate the hawkes_mask)
        event_assignment = tf.argmax(event_assignment_log_prob, output_type=tf.int32)
        hawkes_mask = tf.assign(hawkes_mask[i], event_assignment)

        set_last_noise_ts = lambda: tf.assign(last_noise_timestamp, events_t[i])
        set_last_hawkes_ll = lambda: tf.assign(prev_hawkes_ll, hawkes_ll)
        f = tf.case([(tf.equal(event_assignment, 0), set_last_noise_ts)], default=set_last_hawkes_ll)

        with tf.control_dependencies([mixture_log_prob, f, hawkes_mask]):
            return [tf.add(i, 1), iters]

    i = tf.constant(0, dtype=tf.int32)
    i, _ = tf.while_loop(cond, body, [i, num_events], name="sample_z", parallel_iterations=1)

    with tf.variable_scope("sample_latent_variable", reuse=tf.AUTO_REUSE):
        hawkes_mask = tf.get_variable('hawkes_mask', dtype=tf.int32)
        mixture_log_prob = tf.get_variable('mixture_log_prob', dtype=tf.float32)

    # Invert the Hawkes mast to get Z (noise mask)
    with tf.control_dependencies([i]):
        z = tf.abs(hawkes_mask - 1)
        mixture_log_prob = mixture_log_prob + 0

    return z, mixture_log_prob


def sample_z_exp_only(events_info, exp_sample_rates, cat_sample_prob):
    rv_exp = tfd.Exponential(rate=exp_sample_rates)
    exp_log_prob = rv_exp.log_prob(tf.reshape(events_info, [tf.shape(events_side_info)[0], 1]))

    stacked_cat_probs = tf.log(tf.stack([cat_sample_prob, 1. - cat_sample_prob], name='p_stacked'))

    mixture_log_prob = stacked_cat_probs + exp_log_prob
    z = tf.argmax(mixture_log_prob, axis=1)

    return z, mixture_log_prob


def sample_z_exp_only_no_cat(events_info, exp_sample_rates):
    rv_exp = tfd.Exponential(rate=exp_sample_rates)
    exp_mixture_log_prob = rv_exp.log_prob(tf.reshape(events_info, [tf.shape(events_side_info)[0], 1]))

    z = tf.argmax(exp_mixture_log_prob, axis=1)

    return z, exp_mixture_log_prob


def sample_z_point_process_only_no_cat(events_t,
                                hw_sample_alpha, hw_sample_beta, hw_sample_intensity,
                                poi_sample_intensity):

    num_events = events_t.get_shape()[0]

    rv_hawkes_observations = hwk.Hawkes(hw_sample_intensity,
                                        hw_sample_alpha,
                                        hw_sample_beta,
                                        tf.float32, name="hawkes_observations_rv")

    rv_exp_poisson_inter_arrival = tfd.Exponential(poi_sample_intensity)

    def cond(i, iters):
        return tf.less(i, iters)

    def body(i, iters):
        with tf.variable_scope("sample_latent_variable", reuse=tf.AUTO_REUSE):
            hawkes_mask = tf.get_variable('hawkes_mask', num_events, dtype=tf.int32,
                                          initializer=tf.zeros_initializer())
            prev_hawkes_ll = tf.get_variable('prev_hawkes_ll', [], dtype=tf.float32, initializer=tf.zeros_initializer())

            last_noise_timestamp = tf.get_variable('last_noise_timestamp', [], dtype=tf.float32,
                                                   initializer=tf.zeros_initializer())

            mixture_log_prob = tf.get_variable('mixture_log_prob', [num_events, 2], dtype=tf.float32,
                                               initializer=tf.zeros_initializer())

        # Hawkes log-likelihood
        hawkes_history = tf.boolean_mask(events_t, hawkes_mask)
        hawkes_ll = rv_hawkes_observations.log_likelihood(tf.concat([hawkes_history, [events_t[i]]], axis=0))

        # Poisson log-likelihood
        poisson_ll = rv_exp_poisson_inter_arrival.log_cdf(events_t[i] - last_noise_timestamp)

        event_assignment_log_prob = tf.stack([poisson_ll, hawkes_ll - prev_hawkes_ll], name='point_process_ll')

        # for logging purposes
        mixture_log_prob = tf.assign(mixture_log_prob[i], event_assignment_log_prob)

        # 0 for poisson/noise, 1 for hawkes (order is changed here to accommodate the hawkes_mask)
        event_assignment = tf.argmax(event_assignment_log_prob, output_type=tf.int32)
        hawkes_mask = tf.assign(hawkes_mask[i], event_assignment)

        set_last_noise_ts = lambda: tf.assign(last_noise_timestamp, events_t[i])
        set_last_hawkes_ll = lambda: tf.assign(prev_hawkes_ll, hawkes_ll)
        f = tf.case([(tf.equal(event_assignment, 0), set_last_noise_ts)], default=set_last_hawkes_ll)

        with tf.control_dependencies([mixture_log_prob, f, hawkes_mask]):
            return [tf.add(i, 1), iters]

    i = tf.constant(0, dtype=tf.int32)
    i, _ = tf.while_loop(cond, body, [i, num_events], name="sample_z", parallel_iterations=1)

    with tf.variable_scope("sample_latent_variable", reuse=tf.AUTO_REUSE):
        hawkes_mask = tf.get_variable('hawkes_mask', dtype=tf.int32)
        mixture_log_prob = tf.get_variable('mixture_log_prob', dtype=tf.float32)

    # Invert the Hawkes mast to get Z (noise mask)
    with tf.control_dependencies([i]):
        z = tf.abs(hawkes_mask - 1)
        mixture_log_prob = mixture_log_prob + 0

    return z, mixture_log_prob



def sample_z_point_process_only(events_t,
                                hw_sample_alpha, hw_sample_beta, hw_sample_intensity,
                                poi_sample_intensity,
                                hawkes_cat_prob):

    num_events = events_t.get_shape()[0]

    stacked_cat_probs = tf.log(tf.stack([1. - hawkes_cat_prob, hawkes_cat_prob], name='p_stacked'))

    rv_hawkes_observations = hwk.Hawkes(hw_sample_intensity,
                                        hw_sample_alpha,
                                        hw_sample_beta,
                                        tf.float32, name="hawkes_observations_rv")

    rv_exp_poisson_inter_arrival = tfd.Exponential(poi_sample_intensity)

    def cond(i, iters):
        return tf.less(i, iters)

    def body(i, iters):
        with tf.variable_scope("sample_latent_variable", reuse=tf.AUTO_REUSE):
            hawkes_mask = tf.get_variable('hawkes_mask', num_events, dtype=tf.int32,
                                          initializer=tf.zeros_initializer())
            prev_hawkes_ll = tf.get_variable('prev_hawkes_ll', [], dtype=tf.float32, initializer=tf.zeros_initializer())

            last_noise_timestamp = tf.get_variable('last_noise_timestamp', [], dtype=tf.float32,
                                                   initializer=tf.zeros_initializer())

            mixture_log_prob = tf.get_variable('mixture_log_prob', [num_events, 2], dtype=tf.float32,
                                               initializer=tf.zeros_initializer())

        # Hawkes log-likelihood
        hawkes_history = tf.boolean_mask(events_t, hawkes_mask)
        hawkes_ll = rv_hawkes_observations.log_likelihood(tf.concat([hawkes_history, [events_t[i]]], axis=0))

        # Poisson log-likelihood
        poisson_ll = rv_exp_poisson_inter_arrival.log_cdf(events_t[i] - last_noise_timestamp)

        event_assignment_log_prob = tf.stack([poisson_ll, hawkes_ll - prev_hawkes_ll], name='point_process_ll')
        event_assignment_log_prob = event_assignment_log_prob + stacked_cat_probs

        # for logging purposes
        mixture_log_prob = tf.assign(mixture_log_prob[i], event_assignment_log_prob)

        # 0 for poisson/noise, 1 for hawkes (order is changed here to accommodate the hawkes_mask)
        event_assignment = tf.argmax(event_assignment_log_prob, output_type=tf.int32)
        hawkes_mask = tf.assign(hawkes_mask[i], event_assignment)

        set_last_noise_ts = lambda: tf.assign(last_noise_timestamp, events_t[i])
        set_last_hawkes_ll = lambda: tf.assign(prev_hawkes_ll, hawkes_ll)
        f = tf.case([(tf.equal(event_assignment, 0), set_last_noise_ts)], default=set_last_hawkes_ll)

        with tf.control_dependencies([mixture_log_prob, f, hawkes_mask]):
            return [tf.add(i, 1), iters]

    i = tf.constant(0, dtype=tf.int32)
    i, _ = tf.while_loop(cond, body, [i, num_events], name="sample_z", parallel_iterations=1)

    with tf.variable_scope("sample_latent_variable", reuse=tf.AUTO_REUSE):
        hawkes_mask = tf.get_variable('hawkes_mask', dtype=tf.int32)
        mixture_log_prob = tf.get_variable('mixture_log_prob', dtype=tf.float32)

    # Invert the Hawkes mast to get Z (noise mask)
    with tf.control_dependencies([i]):
        z = tf.abs(hawkes_mask - 1)
        mixture_log_prob = mixture_log_prob + 0

    return z, mixture_log_prob

z, mixture_log_prob = sample_z(event_times, events_side_info,
                               _h_alpha, _h_beta, _h_intensity,
                               _p_intensity,
                               _h_exp_rate, _p_exp_rate,
                               1. - hum.noise_percentage)

# z, mixture_log_prob = sample_z_point_process_only(event_times,
#                                                   _h_alpha, _h_beta, _h_intensity,
#                                                   _p_intensity,
#                                                   1. - hum.noise_percentage)

# z, mixture_log_prob = sample_z_exp_only_no_cat(events_side_info, [_h_exp_rate, _p_exp_rate])
# g = tf.shape(event_times)[0]
# g = g + 1

# rv_exp_poisson_inter_arrival = tfd.Exponential(.5)
# adf = rv_exp_poisson_inter_arrival.cdf(1)
with tf.Session() as sess:
    # print(sess.run(adf))
    sess.run(tf.global_variables_initializer())
    [z_,
     mixture_log_prob_] = sess.run([z,
                                    mixture_log_prob])


print(z_)
print(np.exp(mixture_log_prob_))
print(np.exp(mixture_log_prob_[:, 0]) / np.sum(np.exp(mixture_log_prob_), axis=1))

# with tf.Session() as sess:
#     # print(sess.run([event_times, events_side_info]))
#     # mixture_prob_, z_ = sess.run([mixture_prob, z])
#     # print(mixture_prob_, z_)
#     sess.run(tf.global_variables_initializer())
#     # print(sess.run(i))
#     z_ = sess.run(z)
#     print(z_)
#     z_exp_ = sess.run(event_assignment_exp)
#     print(z_exp_)
#
#     print(sess.run(time_noise))
#     # print(sess.run(exp_mixture_log_prob))
# print(z_)
# print(hum.mixed_timestamps)
print(hum.mixed_labels)
#
# print(f1_score(z_, z_exp_))
#
#
# print(f1_score(hum.mixed_labels, z_exp_))
print(f1_score(hum.mixed_labels, z_))
# # print("exp only:", f1_score(hum.mixed_labels, z_))
# # print("Noise Percentage: ", hum.noise_percentage)