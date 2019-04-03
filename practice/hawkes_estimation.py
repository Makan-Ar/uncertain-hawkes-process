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

event_times = tf.convert_to_tensor(hawkes_event_times, name="event_times_data", dtype=tf.float32)

# rv_test = hwk.Hawkes(_intensity, 0.9, _beta, tf.float32, name="hawkes_observations")
# a = rv_test.log_likelihood(event_times)
# # [-1641056.5, -1043076.0]
#
# rv_test_1 = hwk.Hawkes(_intensity, 9, _beta, tf.float32, name="hawkes_observations_1")
# b = rv_test_1.log_likelihood(event_times)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     sess.run(tf.local_variables_initializer())
#     print(sess.run([a, b]))


def joint_log_prob_alpha(data, sample_alpha):
    # rv_alpha = tfd.Uniform(0., 1., name='alpha_prior_rv')
    rv_alpha = tfd.Exponential(rate=0.01, name='alpha_prior_rv')

    rv_observations = hwk.Hawkes(_intensity, _alpha, sample_alpha, tf.float32, name="hawkes_observations_rv")

    return (
        rv_alpha.log_prob(sample_alpha) +
        rv_observations.log_likelihood(data)
    )


def joint_log_prob_alpha_beta(data, sample_alpha, sample_beta):
    rv_alpha = tfd.Exponential(rate=0.01, name='alpha_prior_rv')
    rv_beta = tfd.Exponential(rate=0.01, name='beta_prior_rv')

    rv_observations = hwk.Hawkes(_intensity, sample_alpha, sample_beta, tf.float32, name="hawkes_observations_rv")

    return (
        rv_alpha.log_prob(sample_alpha) +
        rv_beta.log_prob(sample_beta) +
        rv_observations.log_likelihood(data)
    )


def joint_log_prob_all(data, sample_alpha, sample_beta, sample_intensity):
    rv_alpha = tfd.Exponential(rate=0.01, name='alpha_prior_rv')
    rv_beta = tfd.Exponential(rate=0.01, name='beta_prior_rv')
    rv_intensity = tfd.Exponential(rate=0.01, name='intensity_prior_rv')

    rv_observations = hwk.Hawkes(sample_intensity, sample_alpha, sample_beta, tf.float32, name="hawkes_observations_rv")

    return (
        rv_alpha.log_prob(sample_alpha) +
        rv_beta.log_prob(sample_beta) +
        rv_intensity.log_prob(sample_intensity) +
        rv_observations.log_likelihood(data)
    )

# estimate_param = 'alpha'
# estimate_param = 'alpha_beta'
estimate_param = 'all'

number_of_steps = 2500
burnin = 250

# set the chain's initial state & define closure over our joint_log_prob
if estimate_param == 'alpha':
    initial_chain_state = [
        tf.constant(0.5, name="init_alpha"),
    ]

    unconstraining_bijectors = [
        tfp.bijectors.Identity()
    ]

    unnormalized_posterior_log_prob = lambda *args: joint_log_prob_alpha(event_times, *args)

elif estimate_param == 'alpha_beta':
    initial_chain_state = [
        tf.constant(0.5, name="init_alpha"),
        tf.constant(0.5, name="init_beta"),
    ]

    unconstraining_bijectors = [
        tfp.bijectors.Identity(),
        tfp.bijectors.Identity()
    ]

    unnormalized_posterior_log_prob = lambda *args: joint_log_prob_alpha_beta(event_times, *args)

else:
    initial_chain_state = [
        tf.constant(0.5, name="init_alpha"),
        tf.constant(0.5, name="init_beta"),
        tf.constant(0.5, name="init_intensity"),
    ]

    unconstraining_bijectors = [
        tfp.bijectors.Identity(),
        tfp.bijectors.Identity(),
        tfp.bijectors.Identity()
    ]

    unnormalized_posterior_log_prob = lambda *args: joint_log_prob_all(event_times, *args)


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

if estimate_param == 'alpha':
    [
        posterior_prob_alpha,
    ], kernel_results = tfp.mcmc.sample_chain(
        num_results=number_of_steps,
        num_burnin_steps=burnin,
        current_state=initial_chain_state,
        kernel=hmc
    )

elif estimate_param == 'alpha_beta':
    [
        posterior_prob_alpha,
        posterior_prob_beta,
    ], kernel_results = tfp.mcmc.sample_chain(
        num_results=number_of_steps,
        num_burnin_steps=burnin,
        current_state=initial_chain_state,
        kernel=hmc
    )

else:
    [
        posterior_prob_alpha,
        posterior_prob_beta,
        posterior_prob_intensity,
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

    if estimate_param == 'alpha':
        [
            posterior_prob_alpha_,
            kernel_results_
        ] = sess.run([
            posterior_prob_alpha,
            kernel_results
        ])

    elif estimate_param == 'alpha_beta':
        [
            posterior_prob_alpha_,
            posterior_prob_beta_,
            kernel_results_
        ] = sess.run([
            posterior_prob_alpha,
            posterior_prob_beta,
            kernel_results
        ])

    else:
        [
            posterior_prob_alpha_,
            posterior_prob_beta_,
            posterior_prob_intensity_,
            kernel_results_
        ] = sess.run([
            posterior_prob_alpha,
            posterior_prob_beta,
            posterior_prob_intensity,
            kernel_results
        ])


print(f"MCMC took {(time.time() - start_time)/60:4.2f}m.")

new_step_size_initializer_ = kernel_results_.inner_results.is_accepted.mean()
print("acceptance rate: {}".format(new_step_size_initializer_))
print("final step size: {}".format(kernel_results_.inner_results.extra.step_size_assign[-100:].mean()))
print(f"Alpha. Mean: {np.mean(posterior_prob_alpha_)}, SD: {np.std(posterior_prob_alpha_)}")

if estimate_param == 'alpha_beta' or estimate_param == 'all':
    print(f"beta. Mean: {np.mean(posterior_prob_beta_)}, SD: {np.std(posterior_prob_beta_)}")

if estimate_param == 'all':
    print(f"intensity. Mean: {np.mean(posterior_prob_intensity_)}, SD: {np.std(posterior_prob_intensity_)}")

# Plotting
lw = 1
plt.subplot(311)
plt.plot(posterior_prob_alpha_, lw=lw, c='red',
         label=f"trace of alpha. Mean: {np.mean(posterior_prob_alpha_):4.4f}, SD: {np.std(posterior_prob_alpha_):4.4f}")
plt.title("Traces of unknown parameters")
leg = plt.legend()
leg.get_frame().set_alpha(0.7)

if estimate_param == 'alpha_beta' or estimate_param == 'all':
    plt.subplot(312)
    plt.plot(posterior_prob_beta_, lw=lw, c='blue',
             label=f"trace of beta. Mean: {np.mean(posterior_prob_beta_):4.4f}, SD: {np.std(posterior_prob_beta_):4.4f}")
    plt.legend()

if estimate_param == 'all':
    plt.subplot(313)
    plt.plot(posterior_prob_intensity_, lw=lw, c='green',
             label=f"trace of intensity. Mean: {np.mean(posterior_prob_intensity_):4.4f}, SD: {np.std(posterior_prob_intensity_):4.4f}")
    plt.legend()


plt.xlabel("Steps")
plt.savefig(f'{plot_base_path}/all-hawkes-trace.pdf')
# plt.show()

plt.clf()

plt.title("Posterior of of unknown parameters")
plt.subplot(311)
plt.title(f"Posterior of alpha")
plt.hist(posterior_prob_alpha_, color='red', bins=50, histtype="stepfilled", label="alpha")

if estimate_param == 'alpha_beta' or estimate_param == 'all':
    plt.subplot(312)
    plt.title(f"Posterior of beta")
    plt.hist(posterior_prob_beta_, color='blue', bins=50, histtype="stepfilled", label="beta")

if estimate_param == 'all':
    plt.subplot(313)
    plt.title(f"Posterior of intensity")
    plt.hist(posterior_prob_intensity_, color='green', bins=50, histtype="stepfilled", label="intensity")

plt.tight_layout()
plt.savefig(f'{plot_base_path}/all-hawkes-hist.pdf')
# plt.show()
