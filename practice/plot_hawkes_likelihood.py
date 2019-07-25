import time
import numpy as np
import hawkes as hwk
import tensorflow as tf
from tick.hawkes import SimuHawkesExpKernels
import matplotlib.pyplot as plt
from tick.plot import plot_point_process
from likelihood_utils import hawkes_log_likelihood_numpy

# _intensity = 0.5
# _beta = 2
# _alpha = 0.9 / _beta
#
# # Hawkes simulation
# n_nodes = 1  # dimension of the Hawkes process
# adjacency = _alpha * np.ones((n_nodes, n_nodes))
# decays = _beta * np.ones((n_nodes, n_nodes))
# baseline = _intensity * np.ones(n_nodes)
# hawkes_sim = SimuHawkesExpKernels(adjacency=adjacency, decays=decays, baseline=baseline, verbose=False, seed=435)
#
# run_time = 1000
# hawkes_sim.end_time = run_time
# dt = 0.01
# hawkes_sim.track_intensity(dt)
# hawkes_sim.simulate()
# event_times = hawkes_sim.timestamps[0]
# # print(len(event_times))
# # plot_point_process(hawkes_sim, n_points=5000, t_min=1, max_jumps=200)
# # plt.show()



import sys
sys.path.insert(0, r'/nethome/marastu2/uncertain-hawkes-process')
from hawkes_uncertain_simulator import HawkesUncertainModel

plot_base_path = '/shared/Results/HawkesUncertainEvents/temp'

_h_intensity = 0.5
_h_beta = 2
_h_alpha = 0.9

_runtime = 15

_p_intensity = 0.2

_h_exp_rate = 1.5
_p_exp_rate = 116.5

hum = HawkesUncertainModel(h_lambda=_h_intensity, h_alpha=_h_alpha, h_beta=_h_beta, h_exp_rate=_h_exp_rate,
                           p_lambda=_p_intensity, p_exp_rate=_p_exp_rate,
                           noise_percentage_ub=0.25, run_time=_runtime, delta=0.01, seed=45)

# event_times = tf.convert_to_tensor(hum.hawkes.timestamps[0], name="event_times_data", dtype=tf.float32)
# events_side_info = tf.convert_to_tensor(hum.hawkes_exp, name="event_side_data", dtype=tf.float32)

event_times = tf.convert_to_tensor(hum.mixed_timestamps, name="event_times_data", dtype=tf.float32)
events_side_info = tf.convert_to_tensor(hum.mixed_expo, name="event_side_data", dtype=tf.float32)

hawkes_event_times = hum.mixed_timestamps

def hawkes_intensity(hawkes_event_times, intensity, alpha, beta):
    hawkes_intensity = np.zeros(len(hawkes_event_times))
    for i in range(1, len(hawkes_event_times)):
        hawkes_intensity[i] = np.sum(alpha * np.exp(-1 * beta * (hawkes_event_times[i] - hawkes_event_times[:i])))

    hawkes_intensity = hawkes_intensity + intensity
    return hawkes_intensity


def exponential_dist(lambd, t):
    return lambd * t * np.exp(lambd * t)

for i in range(1, len(hum.poisson.timestamps[0])):
    print(exponential_dist(_p_intensity, hum.poisson.timestamps[0][i] - hum.poisson.timestamps[0][i-1]))

print()

for i in range(1, len(hum.hawkes.timestamps[0])):
    print(exponential_dist(_p_intensity, hum.hawkes.timestamps[0][i] - hum.hawkes.timestamps[0][i-1]))

# plt.plot(hum.hawkes.timestamps[0], hawkes_intensity(hum.hawkes.timestamps[0], _h_intensity, _h_alpha, _h_beta))
# for ht in hum.hawkes.timestamps[0]:
#     plt.axvline(x=ht, color="black")
# plt.show()
#
# plt.plot(hum.poisson.timestamps[0], hawkes_intensity(hum.poisson.timestamps[0], _h_intensity, _h_alpha, _h_beta))
# for ht in hum.poisson.timestamps[0]:
#     plt.axvline(x=ht, color="black")
# plt.show()
#
#
# plt.plot(hum.hawkes.timestamps[0], hawkes_intensity(hum.hawkes.timestamps[0], _h_intensity, _h_alpha, _h_beta), ls='--',
#          color='black', alpha=0.5)
#
# plt.plot(hum.poisson.timestamps[0], hawkes_intensity(hum.poisson.timestamps[0], _h_intensity, _h_alpha, _h_beta), ls=':',
#          color="red", alpha=0.5)
#
# plt.plot(event_times, hawkes_intensity(event_times, _h_intensity, _h_alpha, _h_beta), color='green',
#          alpha=0.5)
# for ht in hum.hawkes.timestamps[0]:
#     plt.axvline(x=ht, color="black", alpha=0.5)
#
# for ht in hum.poisson.timestamps[0]:
#     plt.axvline(x=ht, color="red", alpha=0.5)
# plt.show()

uhll = [0]
hll = [0]
pll = [0]
for i in range(1, len(hawkes_event_times)):
    ll = hawkes_log_likelihood_numpy(hawkes_event_times[:i], _h_intensity, _h_alpha, _h_beta)
    uhll.append(ll)

uhll = np.array(uhll)
uhll[1:] = uhll[1:] - uhll[:-1]

for i in range(1, len(hum.hawkes.timestamps[0])):
    ll = hawkes_log_likelihood_numpy(hum.hawkes.timestamps[0][:i], _h_intensity, _h_alpha, _h_beta)
    hll.append(ll)

hll = np.array(hll)
hll[1:] = hll[1:] - hll[:-1]

for i in range(1, len(hum.poisson.timestamps[0])):
    ll = hawkes_log_likelihood_numpy(hum.poisson.timestamps[0][:i], _h_intensity, _h_alpha, _h_beta)
    pll.append(ll)

pll = np.array(pll)
pll[1:] = pll[1:] - pll[:-1]

plt.plot(hawkes_event_times, uhll, c='red', label="Both")
plt.plot(hum.hawkes.timestamps[0], hll, c='blue', label="Hawkes")
plt.plot(hum.poisson.timestamps[0], pll, c='green', label='Poisson')

for pt in hum.poisson.timestamps[0]:
    plt.axvline(x=pt, color="orange")

for ht in hum.hawkes.timestamps[0]:
    plt.axvline(x=ht, color="black")

plt.legend()
plt.xlabel("Hawkes Arrival Times")
plt.xticks(hawkes_event_times)
# plt.ylim(0, 1)
# plt.legend()
plt.show()


# for i in [10, 2, 1.7,  1.5, 0.9, 0.5, 0.1]:
#     ti = time.time()
#     print(i)
#     res = hawkes_log_likelihood(event_times, _intensity, i, _beta)
#     j = time.time() - ti
#     print(res)

# print()
# result = []
# for i in np.arange(0, 12, 0.1):
#     result.append(hawkes_log_likelihood(event_times, _intensity, i, _beta))
#     print(i, end='\r')
#
# print()
# plt.plot(np.arange(0, 12, 0.1), result, c='red')
# plt.axvline(x=_alpha)
# # plt.xlabel("Steps")
# # plt.ylim(0, 1)
# # plt.legend()
# plt.show()
#
# event_times = tf.convert_to_tensor(event_times, name="event_times_data", dtype=tf.float64)
#
# ti = time.time()
# res = hawkes_log_likelihood(event_times, _intensity, alpha_test, _beta)
# j = time.time() - ti
#
# hawkes = hwk.Hawkes(_intensity, alpha_test, _beta, tf.float64)
# rest = hawkes.log_likelihood(event_times)
#
# with tf.Session() as sess:
#     writer = tf.summary.FileWriter('./graph-files/myg.g', sess.graph)
#     sess.run(tf.global_variables_initializer())
#     sess.run(tf.local_variables_initializer())
#     ti = time.time()
#     print(sess.run(rest))
#     jt = time.time() - ti
#
# writer.close()
# print(rest)
# print(res)
# print(jt)
# print(j)
