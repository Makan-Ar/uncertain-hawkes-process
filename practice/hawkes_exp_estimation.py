import time
import numpy as np
import hawkes as hwk
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from tick.hawkes import SimuHawkesExpKernels

tfd = tfp.distributions

plot_base_path = '/shared/Results/HawkesUncertainEvents/temp'

_intensity = 0.5
_beta = 2
_alpha = 0.9
_runtime = 1000

_exp_rate = 1.5


# Hawkes simulation
_alpha /= _beta
n_nodes = 1  # dimension of the Hawkes process
adjacency = _alpha * np.ones((n_nodes, n_nodes))
decays = _beta * np.ones((n_nodes, n_nodes))
baseline = _intensity * np.ones(n_nodes)
hawkes_sim = SimuHawkesExpKernels(adjacency=adjacency, decays=decays, baseline=baseline, verbose=False, seed=435)

hawkes_sim.end_time = _runtime
dt = 0.01
hawkes_sim.track_intensity(dt)
hawkes_sim.simulate()
hawkes_event_times = hawkes_sim.timestamps[0]

event_times = tf.convert_to_tensor(hawkes_event_times, name="event_times_data", dtype=tf.float32)

# Exp simulation
exp_dist = tfd.Exponential(rate=_exp_rate)
events_side_info = exp_dist.sample(tf.shape(event_times))

# with tf.Session() as sess:
#     print(sess.run([event_times, events_side_info]))


def joint_log_prob(events_t, events_info, sample_alpha, sample_beta, sample_intensity, sample_rate):
    rv_alpha = tfd.Exponential(rate=0.01, name='alpha_prior_rv')
    rv_beta = tfd.Exponential(rate=0.01, name='beta_prior_rv')
    rv_intensity = tfd.Exponential(rate=0.01, name='intensity_prior_rv')

    rv_hawkes_observations = hwk.Hawkes(sample_intensity,
                                        sample_alpha,
                                        sample_beta,
                                        tf.float32, name="hawkes_observations_rv")

    # rv_rate
    rv_uniform = tfd.Uniform(0.0001, 100, name="rate_prior_rv")
    rv_exp_observation = tfd.Exponential(rate=sample_rate, name="exp_observations_rv")

    return (
        rv_alpha.log_prob(sample_alpha) +
        rv_beta.log_prob(sample_beta) +
        rv_intensity.log_prob(sample_intensity) +
        rv_hawkes_observations.log_likelihood(events_t) +

        rv_uniform.log_prob(sample_rate) +
        tf.reduce_sum(rv_exp_observation.log_prob(events_info))
    )


number_of_steps = 2500
burnin = 250

# set the chain's initial state & define closure over our joint_log_prob
initial_chain_state = [
    tf.constant(0.5, name="init_alpha"),
    tf.constant(0.5, name="init_beta"),
    tf.constant(0.5, name="init_intensity"),
    tf.constant(0.5, name='init_rate'),
]

unconstraining_bijectors = [
    tfp.bijectors.Identity(),
    tfp.bijectors.Identity(),
    tfp.bijectors.Identity(),
    tfp.bijectors.Identity()
]

unnormalized_posterior_log_prob = lambda *args: joint_log_prob(event_times, events_side_info, *args)


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
    posterior_prob_alpha,
    posterior_prob_beta,
    posterior_prob_intensity,
    posterior_prob_rate
], kernel_results = tfp.mcmc.sample_chain(
    num_results=number_of_steps,
    num_burnin_steps=burnin,
    current_state=initial_chain_state,
    kernel=hmc
)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

tic = time.time()
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    [
        posterior_prob_alpha_,
        posterior_prob_beta_,
        posterior_prob_intensity_,
        posterior_prob_rate_,
        kernel_results_
    ] = sess.run([
        posterior_prob_alpha,
        posterior_prob_beta,
        posterior_prob_intensity,
        posterior_prob_rate,
        kernel_results
    ])
toc = time.time()

print(f"MCMC took {(toc - tic)/60:4.2f}m.")

new_step_size_initializer_ = kernel_results_.inner_results.is_accepted.mean()
print("acceptance rate: {}".format(new_step_size_initializer_))
print("final step size: {}".format(kernel_results_.inner_results.extra.step_size_assign[-100:].mean()))
print(f"Alpha. Mean: {np.mean(posterior_prob_alpha_)}, SD: {np.std(posterior_prob_alpha_)}")
print(f"beta. Mean: {np.mean(posterior_prob_beta_)}, SD: {np.std(posterior_prob_beta_)}")
print(f"intensity. Mean: {np.mean(posterior_prob_intensity_)}, SD: {np.std(posterior_prob_intensity_)}")
print(f"rate. Mean: {np.mean(posterior_prob_rate_)}, SD: {np.std(posterior_prob_rate_)}")

# Plotting
lw = 1
plt.subplot(411)
plt.plot(posterior_prob_alpha_, lw=lw, c='red',
         label=f"trace of alpha. Mean: {np.mean(posterior_prob_alpha_):4.4f}, SD: {np.std(posterior_prob_alpha_):4.4f}")
plt.title("Traces of unknown parameters")
leg = plt.legend()
leg.get_frame().set_alpha(0.7)

plt.subplot(412)
plt.plot(posterior_prob_beta_, lw=lw, c='blue',
         label=f"trace of beta. Mean: {np.mean(posterior_prob_beta_):4.4f}, SD: {np.std(posterior_prob_beta_):4.4f}")
plt.legend()

plt.subplot(413)
plt.plot(posterior_prob_intensity_, lw=lw, c='green',
         label=f"trace of intensity. Mean: {np.mean(posterior_prob_intensity_):4.4f}, "
         f"SD: {np.std(posterior_prob_intensity_):4.4f}")
plt.legend()

plt.subplot(414)
plt.plot(posterior_prob_rate_, lw=lw, c='gray',
         label=f"trace of rate. Mean: {np.mean(posterior_prob_rate_):4.4f}, SD: {np.std(posterior_prob_rate_):4.4f}")
plt.legend()


plt.xlabel("Steps")
plt.savefig(f'{plot_base_path}/hawkes-exp-trace.pdf')
plt.show()

plt.clf()

plt.title("Posterior of of unknown parameters")
plt.subplot(411)
plt.title(f"Posterior of alpha")
plt.hist(posterior_prob_alpha_, color='red', bins=50, histtype="stepfilled", label="alpha")

plt.subplot(412)
plt.title(f"Posterior of beta")
plt.hist(posterior_prob_beta_, color='blue', bins=50, histtype="stepfilled", label="beta")

plt.subplot(413)
plt.title(f"Posterior of intensity")
plt.hist(posterior_prob_intensity_, color='green', bins=50, histtype="stepfilled", label="intensity")

plt.subplot(414)
plt.title(f"Posterior of rate")
plt.hist(posterior_prob_rate_, color='gray', bins=50, histtype="stepfilled", label="rate")

plt.tight_layout()
plt.savefig(f'{plot_base_path}/hawkes-exp-hist.pdf')
plt.show()
