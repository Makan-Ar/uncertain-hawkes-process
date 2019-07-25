import time
import numpy as np
import hawkes as hwk
import tensorflow as tf
import matplotlib.pyplot as plt
from tick.plot import plot_point_process
from tick.hawkes import SimuHawkesExpKernels
from likelihood_utils import hawkes_log_likelihood_numpy



_intensity = 0.5
_beta = 2
_alpha = 0.9 / _beta

# Hawkes simulation
n_nodes = 1  # dimension of the Hawkes process
adjacency = _alpha * np.ones((n_nodes, n_nodes))
decays = _beta * np.ones((n_nodes, n_nodes))
baseline = _intensity * np.ones(n_nodes)
hawkes_sim = SimuHawkesExpKernels(adjacency=adjacency, decays=decays, baseline=baseline, verbose=False, seed=435)

run_time = 1000
hawkes_sim.end_time = run_time
dt = 0.01
hawkes_sim.track_intensity(dt)
hawkes_sim.simulate()
hawkes_event_times = hawkes_sim.timestamps[0]
# print(len(event_times))
# plot_point_process(hawkes_sim, n_points=5000, t_min=1, max_jumps=200)
# plt.show()


event_times_loaded = np.loadtxt('./hsim.csv', skiprows=1)
hawkes_event_times = event_times_loaded.ravel()
print(len(hawkes_event_times))

# # This is a bad implementation!
# def hawkes_cum_log_likelihood(hawkes_event_times, intensity, alpha, beta):
#     a_calc = np.zeros(len(hawkes_event_times))
#     for i in range(1, len(hawkes_event_times)):
#         a_calc[i] = np.exp(-1 * beta * (hawkes_event_times[i] - hawkes_event_times[i - 1])) * (1 + a_calc[i - 1])
#
#     term1 = np.sum(np.log(intensity + alpha * a_calc))
#
#     term2 = np.sum(intensity * hawkes_event_times)
#
#     ker_ = 0
#     for k in range(1, len(hawkes_event_times)):
#         ker_ += np.sum(np.exp(-1 * beta * (hawkes_event_times[k] - hawkes_event_times[0:k])) - 1)
#     term3 = (alpha / beta) * ker_
#
#     res = term1 - term2 + term3
#     return res
#     # return term1, -1 * term2, term3, res

alpha_test = 0.9

# for i in [10, 2, 1.7,  1.5, 0.9, 0.5, 0.1]:
#     ti = time.time()
#     print(i)
#     res = hawkes_log_likelihood_numpy(event_times, _intensity, i, _beta)
#     j = time.time() - ti
#     print(res)

# print()
# result = []
# for i in np.arange(0, 12, 0.1):
#     result.append(hawkes_log_likelihood_numpy(event_times, _intensity, i, _beta))
#     print(i, end='\r')
#
# print()
# plt.plot(np.arange(0, 12, 0.1), result, c='red')
# plt.axvline(x=_alpha)
# # plt.xlabel("Steps")
# # plt.ylim(0, 1)
# # plt.legend()
# plt.show()

event_times = tf.convert_to_tensor(hawkes_event_times, name="event_times_data", dtype=tf.float64)

ti = time.time()
res = hawkes_log_likelihood_numpy(hawkes_event_times, _intensity, alpha_test, _beta)
j = time.time() - ti

hawkes = hwk.Hawkes(_intensity, alpha_test, _beta, tf.float64)
rest = hawkes.log_likelihood(event_times)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graph-files/myg.g', sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    ti = time.time()
    print(sess.run(rest))
    jt = time.time() - ti

writer.close()
print(rest)
print(res)
print(jt)
print(j)
