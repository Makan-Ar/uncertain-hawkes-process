import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

tfd = tfp.distributions

rate_1 = tf.constant(1./2., name="rate_1")
rate_2 = tf.constant(1./6., name="rate_2")

data_generator_1 = tfd.Exponential(rate=rate_1)
data_generator_2 = tfd.Exponential(rate=rate_2)

sample_1 = data_generator_1.sample(200)
sample_2 = data_generator_2.sample(250)

dataset = tf.concat(values=[sample_1, sample_2], axis=0, name="dataset")

# with tf.Session() as sess:
#     sample_1_, sample_2_, dataset_ = sess.run([sample_1, sample_2, dataset])
#
#
# plt.hist(sample_1_, bins=30, color="k", histtype="stepfilled", alpha=0.8)
# plt.hist(sample_2_, bins=30, color="g", histtype="stepfilled", alpha=0.8)
# plt.title("Histogram of the dataset")
# plt.show()
#
# plt.clf()
# plt.hist(dataset_, bins=30, color="k", histtype="stepfilled", alpha=0.8)
# plt.title("Histogram of the dataset")
# plt.show()


def joint_log_prob(dataset, sample_prob_1, sample_rates):
    rv_prob = tfd.Uniform(0., 1., name='p')

    sample_prob_2 = tf.subtract(1., sample_prob_1, name='p2')
    stacked_p_rv = tf.stack([sample_prob_1, sample_prob_2], name='p_stacked')
    rv_assignments = tfd.Categorical(probs=stacked_p_rv, name='assignments')

    rv_rates = tfd.Uniform(low=[0., 0.], high=[1., 1.], name="rates_prior_rv")
    rv_observation = tfd.MixtureSameFamily(
        mixture_distribution=rv_assignments,
        components_distribution=tfd.Exponential(sample_rates)
    )

    return (
        rv_prob.log_prob(sample_prob_1) +
        rv_prob.log_prob(sample_prob_2) +
        tf.reduce_sum(rv_rates.log_prob(sample_rates)) +
        tf.reduce_sum(rv_observation.log_prob(dataset))
    )


# MCMC
number_of_steps = 25000
burnin = 1000

# set the chain's initial state
initial_chain_state = [
    tf.constant(0.5, name="init_prob"),
    tf.constant([1./3., 1./4.], name="init_rates")
]

unconstraining_bijectors = [
    tfp.bijectors.Identity(),
    tfp.bijectors.Identity()
]

# define closure over our joint_log_prob
unnormalized_posterior_log_prob = lambda *args: joint_log_prob(dataset, *args)

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
    posterior_prob,
    posterior_rates
], kernel_results = tfp.mcmc.sample_chain(
    num_results=number_of_steps,
    num_burnin_steps=burnin,
    current_state=initial_chain_state,
    kernel=hmc
)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    [
        posterior_prob_,
        posterior_rates_,
        kernel_results_
    ] = sess.run([
        posterior_prob,
        posterior_rates,
        kernel_results
    ])


new_step_size_initializer_ = kernel_results_.inner_results.is_accepted.mean()
print("acceptance rate: {}".format(new_step_size_initializer_))
print("final step size: {}".format(kernel_results_.inner_results.extra.step_size_assign[-100:].mean()))


# Plotting
lw = 1

plt.subplot(211)
plt.plot(1 / posterior_rates_[:, 0], label=f"trace of rate 0. Mean: {1 / np.mean(posterior_rates_[:, 0])}", c='red', lw=lw)
plt.plot(1 / posterior_rates_[:, 1], label=f"trace of rate 1. Mean: {1 / np.mean(posterior_rates_[:, 1])}", c='blue', lw=lw)
plt.title("Traces of unknown parameters")
leg = plt.legend(loc="upper right")
leg.get_frame().set_alpha(0.7)

plt.subplot(212)
plt.plot(posterior_prob_, label=f"$p$: frequency of assignment to cluster 0. Mean: {np.mean(posterior_prob_)}",
         color='green', lw=lw)
plt.xlabel("Steps")
plt.ylim(0, 1)
plt.legend()
plt.show()

plt.clf()

plt.subplot(211)
plt.title(f"Posterior of rate of 0")
plt.hist(1 / posterior_rates_[:, 0], color='red', bins=30, histtype="stepfilled")

plt.subplot(212)
plt.title(f"Posterior of rate of 1")
plt.hist(1 / posterior_rates_[:, 1], color='blue', bins=30, histtype="stepfilled")

plt.tight_layout()
plt.show()

