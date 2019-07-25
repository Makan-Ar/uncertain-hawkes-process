import time
import numpy as np
import hawkes as hwk
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from tick.hawkes import SimuHawkesExpKernels
from hawkes_uncertain_simulator import HawkesUncertainModel

tfd = tfp.distributions

plot_base_path = '/shared/Results/HawkesUncertainEvents/temp'

_h_intensity = 0.5
_h_beta = 2
_h_alpha = 0.9

_runtime = 40

_p_intensity = 0.2

_h_exp_rate = 1.5
_p_exp_rate = 116.5

hum = HawkesUncertainModel(h_lambda=_h_intensity, h_alpha=_h_alpha, h_beta=_h_beta, h_exp_rate=_h_exp_rate,
                           p_lambda=_p_intensity, p_exp_rate=_p_exp_rate,
                           noise_percentage_ub=0.25, run_time=_runtime, delta=0.01, seed=435)

# event_times = tf.convert_to_tensor(hum.hawkes.timestamps[0], name="event_times_data", dtype=tf.float32)
# events_side_info = tf.convert_to_tensor(hum.hawkes_exp, name="event_side_data", dtype=tf.float32)

event_times = tf.convert_to_tensor(hum.mixed_timestamps, name="event_times_data", dtype=tf.float32)
events_side_info = tf.convert_to_tensor(hum.mixed_expo, name="event_side_data", dtype=tf.float32)


with tf.Session() as sess:
    print(sess.run([event_times, events_side_info]))

print("Noise Percentage: ", hum.noise_percentage)

exit()

# def sample_z(events_t, events_info,
#                    hw_sample_alpha, hw_sample_beta, hw_sample_intensity,
#                    exp_sample_rates,
#                    cat_sample_prob):


def joint_log_prob(events_t, events_info,
                   hw_sample_alpha, hw_sample_beta, hw_sample_intensity,
                   exp_sample_rates,
                   cat_sample_prob):
    # Clustering
    # Here we assume exp_sample_rates[1] to be the hawkes processes side info exp rate
    exp_dist_clustering = tfd.Exponential(rate=exp_sample_rates)
    cluster_prob_assignment = exp_dist_clustering.log_prob(tf.reshape(events_info, [tf.size(events_info), 1]))
    hawkes_cluster_ind = tf.math.argmax(cluster_prob_assignment, axis=1, output_type=tf.dtypes.int32)
    num_hawkes_events = tf.reduce_sum(hawkes_cluster_ind)
    events_t = tf.boolean_mask(events_t, hawkes_cluster_ind, name='boolean_mask', axis=None)

    # Hawkes
    rv_alpha = tfd.Exponential(rate=0.01, name='alpha_prior_rv')
    rv_beta = tfd.Exponential(rate=0.01, name='beta_prior_rv')
    rv_intensity = tfd.Exponential(rate=0.01, name='intensity_prior_rv')

    rv_hawkes_observations = hwk.Hawkes(hw_sample_intensity,
                                        hw_sample_alpha,
                                        hw_sample_beta,
                                        tf.float32, name="hawkes_observations_rv")

    default_neg = lambda: tf.constant(0., dtype=tf.float32)
    hll = lambda: rv_hawkes_observations.log_likelihood(events_t)
    hawkes_log_likelihood = tf.case([(tf.less(num_hawkes_events, 1), default_neg)], default=hll)

    # exp mixture
    rv_pi = tfd.Dirichlet([0.5, 0.5], name='pi')

    sample_prob_2 = tf.subtract(1., cat_sample_prob)
    stacked_p_rv = tf.stack([cat_sample_prob, sample_prob_2], name='p_stacked')

    rv_assignments = tfd.Categorical(probs=stacked_p_rv, name='assignments')

    # rv_rates = tfd.Uniform(low=[0.001, 0.001], high=[100., 100.], name="rates_prior_rv")
    rv_rates = tfd.Gamma(concentration=[0.001, 0.001], rate=[0.001, 0.001], name="rates_prior_rv")
    rv_observation = tfd.MixtureSameFamily(
        mixture_distribution=rv_assignments,
        components_distribution=tfd.Exponential(rate=exp_sample_rates)
    )

    return (
        rv_alpha.log_prob(hw_sample_alpha) +
        rv_beta.log_prob(hw_sample_beta) +
        rv_intensity.log_prob(hw_sample_intensity) +
        hawkes_log_likelihood +

        rv_pi.log_prob(stacked_p_rv) +
        tf.reduce_sum(rv_rates.log_prob(exp_sample_rates)) +
        tf.reduce_sum(rv_observation.log_prob(events_info))
    )


# def joint_log_prob(events_t, events_info, sample_alpha, sample_beta, sample_intensity, sample_rate):
#     rv_alpha = tfd.Exponential(rate=0.01, name='alpha_prior_rv')
#     rv_beta = tfd.Exponential(rate=0.01, name='beta_prior_rv')
#     rv_intensity = tfd.Exponential(rate=0.01, name='intensity_prior_rv')
#
#     rv_hawkes_observations = hwk.Hawkes(sample_intensity,
#                                         sample_alpha,
#                                         sample_beta,
#                                         tf.float32, name="hawkes_observations_rv")
#
#     # rv_rate
#     rv_uniform = tfd.Uniform(0.0001, 100, name="rate_prior_rv")
#     rv_exp_observation = tfd.Exponential(rate=sample_rate, name="exp_observations_rv")
#
#     return (
#         rv_alpha.log_prob(sample_alpha) +
#         rv_beta.log_prob(sample_beta) +
#         rv_intensity.log_prob(sample_intensity) +
#         rv_hawkes_observations.log_likelihood(events_t) +
#
#         rv_uniform.log_prob(sample_rate) +
#         tf.reduce_sum(rv_exp_observation.log_prob(events_info))
#     )


number_of_steps = 250
burnin = 25

# set the chain's initial state & define closure over our joint_log_prob
initial_chain_state = [
    tf.constant(0.5, name="init_alpha"),
    tf.constant(0.5, name="init_beta"),
    tf.constant(0.5, name="init_intensity"),
    # tf.constant([np.mean(hum.mixed_expo[:len(hum.mixed_expo)]), np.mean(hum.mixed_expo[len(hum.mixed_expo):])],
    #             name='init_rates'),
    tf.constant([116.5, 1.5], name='init_rates'),
    tf.constant(0.5, name="init_cat_prob"),
]

unconstraining_bijectors = [
    tfp.bijectors.Identity(),
    tfp.bijectors.Identity(),
    tfp.bijectors.Identity(),
    tfp.bijectors.Identity(),
    tfp.bijectors.Identity(),
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
    posterior_prob_rates,
    posterior_prob_cat,
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
        posterior_prob_rates_,
        posterior_prob_cat_,
        kernel_results_
    ] = sess.run([
        posterior_prob_alpha,
        posterior_prob_beta,
        posterior_prob_intensity,
        posterior_prob_rates,
        posterior_prob_cat,
        kernel_results
    ])
toc = time.time()

print(f"MCMC took {(toc - tic)/60:4.2f}m.")

new_step_size_initializer_ = kernel_results_.inner_results.is_accepted.mean()
print("acceptance rate: {}".format(new_step_size_initializer_))
print("final step size: {}".format(kernel_results_.inner_results.extra.step_size_assign[-100:].mean()))
print(f"\nAlpha. Mean: {np.mean(posterior_prob_alpha_):4.4f}, SD: {np.std(posterior_prob_alpha_):4.4f}")
print(f"beta. Mean: {np.mean(posterior_prob_beta_):4.4f}, SD: {np.std(posterior_prob_beta_):4.4f}")
print(f"intensity. Mean: {np.mean(posterior_prob_intensity_):4.4f}, SD: {np.std(posterior_prob_intensity_):4.4f}")
print(f"\nHawkes exp rate. Mean: {np.mean(posterior_prob_rates_[:, 1]):4.4f}, "
      f"SD: {np.std(posterior_prob_rates_[:, 1])}:4.4f")
print(f"Poisson exp rate. Mean: {np.mean(posterior_prob_rates_[:, 0]):4.4f}, "
      f"SD: {np.std(posterior_prob_rates_[:, 0])}:4.4f")
print(f"\nCategorical prob mean. Mean: {np.mean(posterior_prob_cat_):4.4f}, SD: {np.std(posterior_prob_cat_)}:4.4f")

# Plotting Hawkes estimated params
lw = 1
plt.subplot(311)
plt.plot(posterior_prob_alpha_, lw=lw, c='red',
         label=f"trace of alpha. Mean: {np.mean(posterior_prob_alpha_):4.4f}, SD: {np.std(posterior_prob_alpha_):4.4f}")
plt.hlines(_h_alpha)

plt.title("Traces of unknown parameters")
leg = plt.legend()
leg.get_frame().set_alpha(0.7)

plt.subplot(312)
plt.plot(posterior_prob_beta_, lw=lw, c='blue',
         label=f"trace of beta. Mean: {np.mean(posterior_prob_beta_):4.4f}, SD: {np.std(posterior_prob_beta_):4.4f}")
plt.hlines(_h_beta)
plt.legend()

plt.subplot(313)
plt.plot(posterior_prob_intensity_, lw=lw, c='green',
         label=f"trace of intensity. Mean: {np.mean(posterior_prob_intensity_):4.4f}, "
         f"SD: {np.std(posterior_prob_intensity_):4.4f}")
plt.hlines(_h_intensity)

plt.legend()
plt.xlabel("Steps")
plt.tight_layout()
plt.savefig(f'{plot_base_path}/full-hawkes-trace.pdf')
plt.show()

plt.clf()

# Plotting exp and categorical estimated params
plt.subplot(211)
plt.plot(posterior_prob_rates_[:, 1], lw=lw, c='purple',
         label=f"trace of Hawkes exp rate 0. Mean: {np.mean(posterior_prob_rates_[:, 1])}, "
         f"SD: {np.std(posterior_prob_rates_[:, 1])}")
plt.hlines(_h_exp_rate)

plt.plot(posterior_prob_rates_[:, 0], lw=lw, c='pink',
         label=f"trace of poisson exp rate 1. Mean: {np.mean(posterior_prob_rates_[:, 0])}, "
         f"SD: {np.std(posterior_prob_rates_[:, 0])}")
plt.hlines(_p_exp_rate)

plt.title("Traces of unknown parameters")
leg = plt.legend(loc="upper right")
leg.get_frame().set_alpha(0.7)

plt.subplot(212)
plt.plot(posterior_prob_cat, label=f"$p$: frequency of noise (poisson). Mean: {np.mean(posterior_prob_cat)}",
         color='gray', lw=lw)
plt.hlines(hum.noise_percentage)

plt.xlabel("Steps")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig(f'{plot_base_path}/full-exp-trace.pdf')
plt.show()

plt.clf()

# Plotting hawkes estimated sample dists hist
plt.title("Posterior of of unknown parameters")
plt.subplot(311)
plt.title(f"Posterior of alpha")
plt.hist(posterior_prob_alpha_, color='red', bins=50, histtype="stepfilled", label="alpha")
plt.vlines(_h_alpha)

plt.subplot(312)
plt.title(f"Posterior of beta")
plt.hist(posterior_prob_beta_, color='blue', bins=50, histtype="stepfilled", label="beta")
plt.vlines(_h_beta)

plt.subplot(313)
plt.title(f"Posterior of intensity")
plt.hist(posterior_prob_intensity_, color='green', bins=50, histtype="stepfilled", label="intensity")
plt.vlines(_h_intensity)

plt.tight_layout()
plt.savefig(f'{plot_base_path}/full-hawkes-hist.pdf')
plt.show()

plt.clf()

# Plotting exp estimated sample dists hist
plt.subplot(311)
plt.title(f"Posterior of Hawkes rate")
plt.hist(posterior_prob_rates_[:, 1], color='purple', bins=50, histtype="stepfilled")
plt.vlines(_h_exp_rate)

plt.subplot(312)
plt.title(f"Posterior of Poisson rate")
plt.hist(posterior_prob_rates_[:, 0], color='pink', bins=50, histtype="stepfilled")
plt.vlines(_p_exp_rate)

plt.subplot(312)
plt.title(f"Posterior of categorical prob")
plt.hist(posterior_prob_cat, color='gray', bins=50, histtype="stepfilled")
plt.vlines(hum.noise_percentage)

plt.tight_layout()
plt.savefig(f'{plot_base_path}/full-exp-hist.pdf')
plt.show()
