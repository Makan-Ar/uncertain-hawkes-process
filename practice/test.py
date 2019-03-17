import time
import numpy as np
import hawkes as hwk
import tensorflow as tf
from tick.hawkes import SimuHawkesExpKernels

# Hawkes simulation
n_nodes = 1  # dimension of the Hawkes process
adjacency = 0.2 * np.ones((n_nodes, n_nodes))
decays = 3 * np.ones((n_nodes, n_nodes))
baseline = 0.5 * np.ones(n_nodes)
hawkes_sim = SimuHawkesExpKernels(adjacency=adjacency, decays=decays, baseline=baseline, verbose=False)

run_time = 10000
hawkes_sim.end_time = run_time
dt = 0.01
hawkes_sim.track_intensity(dt)
hawkes_sim.simulate()
hawkes_event_times = np.float32(hawkes_sim.timestamps[0])

intensity = 0.5
alpha = 0.2
beta = 3

ti = time.time()
a_calc = np.zeros(len(hawkes_event_times))
for i in range(1, len(hawkes_event_times)):
    a_calc[i] = np.exp(-1 * beta * hawkes_event_times[i] - hawkes_event_times[i - 1]) * (1 + a_calc[i - 1])
term1 = np.sum(np.log(intensity + alpha * a_calc))

term2 = np.sum(intensity * hawkes_event_times)

# ker = 0
# for k in range(0, len(hawkes_event_times)):
#     temp_ker = 0
#     for i in range(k + 1):
#         temp_ker += np.exp(-1 * beta * (hawkes_event_times[k] - hawkes_event_times[i])) - 1
#     ker += temp_ker
# term3 = (alpha / beta) * ker

ker_ = 0
for k in range(1, len(hawkes_event_times)):
    ker_ += np.sum(np.exp(-1 * beta * (hawkes_event_times[k] - hawkes_event_times[0:k])) - 1)
term3 = (alpha / beta) * ker_


res = term1 - term2 + term3
j = time.time() - ti

hawkes = hwk.Hawkes(hawkes_event_times, intensity, alpha, beta)
term1t, term2t, term3t, rest = hawkes.cum_log_likelihood()
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graph-files/myg.g', sess.graph)
    sess.run(tf.global_variables_initializer())
    ti = time.time()
    print(sess.run([term1t, term2t, term3t, rest]))
    # sess.run(a)
    jt = time.time() - ti

writer.close()
print(term1t, term2t, term3t, rest)
print(term1, term2, term3, res)
print(jt)
print(j)

# print(a_calc)