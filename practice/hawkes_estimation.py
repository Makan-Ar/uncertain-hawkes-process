import time
import numpy as np
import hawkes as hwk
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from tick.hawkes import SimuHawkesExpKernels

tfd = tfp.distributions


_intensity = 0.5
_alpha = 0.9
_beta = 10

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


event_times = tf.convert_to_tensor(hawkes_event_times, name="event_times_data", dtype=tf.float32)

# rv_test = hwk.Hawkes(event_times, _intensity, 0.9, _beta, tf.float32, name="hawkes_observations")
# a = rv_test.log_likelihood()
# # [-1641056.5, -1043076.0]
#
# rv_test_1 = hwk.Hawkes(event_times, _intensity, 0.01, _beta, tf.float32, name="hawkes_observations_1")
# b = rv_test_1.log_likelihood()
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     sess.run(tf.local_variables_initializer())
#     print(sess.run([a, b]))

def joint_log_prob(data, sample_alpha):
    # rv_alpha = tfd.Uniform(0., 1., name='alpha_prior_rv')
    rv_alpha = tfd.Exponential(rate=0.01, name='alpha_prior_rv')

    rv_observations = hwk.Hawkes(data, _intensity, sample_alpha, _beta, tf.float32, name="hawkes_observations_rv")

    return (
        rv_alpha.log_prob(sample_alpha) +
        rv_observations.log_likelihood()
    )

number_of_steps = 2500
burnin = 250

# set the chain's initial state
initial_chain_state = [
    tf.constant(0.5, name="init_alpha"),
]

unconstraining_bijectors = [
    tfp.bijectors.Identity()
]

# define closure over our joint_log_prob
unnormalized_posterior_log_prob = lambda *args: joint_log_prob(event_times, *args)

# init the step size
with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    step_size = tf.get_variable(
        name='step_size',
        initializer=tf.constant(0.5, dtype=tf.float32),
        trainable=False,
        use_resource=True
    )

hmc = tfp.mcmc.TransformedTransitionKernel(
    inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=unnormalized_posterior_log_prob,
        num_leapfrog_steps=2,
        step_size=step_size,
        step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(None),
        state_gradients_are_stopped=True
    ),
    bijector=unconstraining_bijectors
)

[
    posterior_prob_alpha
], kernel_results = tfp.mcmc.sample_chain(
    num_results=number_of_steps,
    num_burnin_steps=burnin,
    current_state=initial_chain_state,
    kernel=hmc
)

start_time = time.time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    [
        posterior_prob_alpha_,
        kernel_results_
    ] = sess.run([
        posterior_prob_alpha,
        kernel_results
    ])

print(f"MCMC took {(time.time() - start_time)/60:4.2f}m.")

new_step_size_initializer_ = kernel_results_.inner_results.is_accepted.mean()
print("acceptance rate: {}".format(new_step_size_initializer_))
print("final step size: {}".format(kernel_results_.inner_results.extra.step_size_assign[-100:].mean()))
print(f"Alpha. Mean: {np.mean(posterior_prob_alpha_)}, SD: {np.std(posterior_prob_alpha_)}")

# Plotting
lw = 1
plt.plot(posterior_prob_alpha_, lw=lw, c='red',
         label=f"trace of alpha. Mean: {np.mean(posterior_prob_alpha_):4.4f}, SD: {np.std(posterior_prob_alpha_):4.4f}")
plt.title("Traces of unknown parameters")
leg = plt.legend(loc="upper right")
leg.get_frame().set_alpha(0.7)
plt.xlabel("Steps")
plt.ylim(0, 1)
plt.legend()
plt.savefig('./plots/alpha-hawkes-trace.py')
# plt.show()

plt.clf()

plt.title("Posterior of alpha")
plt.hist(posterior_prob_alpha_, color='red', bins=50, histtype="stepfilled")

plt.tight_layout()
plt.savefig('./plots/alpha-hawkes-hist.py')
# plt.show()
