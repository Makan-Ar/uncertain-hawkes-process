import time
import numpy as np
import tensorflow as tf
from sklearn import metrics
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

tfd = tfp.distributions

rate_1 = tf.constant(1./3., name="rate_1")
rate_2 = tf.constant(1./2., name="rate_2")

data_generator_1 = tfd.Exponential(rate=rate_1)
data_generator_2 = tfd.Exponential(rate=rate_2)

sample_1 = data_generator_1.sample(2000)
sample_2 = data_generator_2.sample(2500)

dataset = tf.concat(values=[sample_1, sample_2], axis=0, name="dataset")
labels = tf.concat(values=[tf.zeros_like(sample_1), tf.ones_like(sample_2)], axis=0, name="labels")

dataset_mean = tf.reduce_mean(dataset)
with tf.Session() as sess:
    dataset_mean_ = sess.run(dataset_mean)
# with tf.Session() as sess:
#     sample_1_, sample_2_, dataset_, dataset_mean_ = sess.run([sample_1, sample_2, dataset, dataset_mean])
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


def joint_log_prob_dirichlet(data, sample_prob_1, sample_rates):
    rv_pi = tfd.Dirichlet([0.5, 0.5], name='pi')

    sample_prob_2 = tf.subtract(1., sample_prob_1)
    stacked_p_rv = tf.stack([sample_prob_1, sample_prob_2], name='p_stacked')

    rv_assignments = tfd.Categorical(probs=stacked_p_rv, name='assignments')

    # rv_rates = tfd.Uniform(low=[0.001, 0.001], high=[100., 100.], name="rates_prior_rv")
    rv_rates = tfd.Gamma(concentration=[0.001, 0.001], rate=[0.001, 0.001], name="rates_prior_rv")
    rv_observation = tfd.MixtureSameFamily(
        mixture_distribution=rv_assignments,
        components_distribution=tfd.Exponential(rate=1 / sample_rates)
    )

    return (
        rv_pi.log_prob(stacked_p_rv) +
        tf.reduce_sum(rv_rates.log_prob(sample_rates)) +
        tf.reduce_sum(rv_observation.log_prob(data))
    )


def joint_log_prob_with_uniform_priors(data, sample_prob_1, sample_rates):
    rv_prob = tfd.Uniform(0., 1., name='p')

    sample_prob_2 = tf.subtract(1., sample_prob_1, name='p2')
    stacked_p_rv = tf.stack([sample_prob_1, sample_prob_2], name='p_stacked')
    rv_assignments = tfd.Categorical(probs=stacked_p_rv, name='assignments')

    rv_rates = tfd.Uniform(low=[0.001, 0.001], high=[100., 100.], name="rates_prior_rv")
    rv_observation = tfd.MixtureSameFamily(
        mixture_distribution=rv_assignments,
        components_distribution=tfd.Exponential(rate=1 / sample_rates)
    )

    return (
        rv_prob.log_prob(sample_prob_1) +
        rv_prob.log_prob(sample_prob_2) +
        tf.reduce_sum(rv_rates.log_prob(sample_rates)) +
        tf.reduce_sum(rv_observation.log_prob(data))
    )


def joint_log_prob_with_gamma_priors(data, sample_prob_1, sample_rates):
    rv_prob = tfd.Uniform(0., 1., name='p')

    sample_prob_2 = tf.subtract(1., sample_prob_1, name='p2')
    stacked_p_rv = tf.stack([sample_prob_1, sample_prob_2], name='p_stacked')
    rv_assignments = tfd.Categorical(probs=stacked_p_rv, name='assignments')

    rv_rates = tfd.Gamma(concentration=[0.001, 0.001], rate=[0.001, 0.001], name="rates_prior_rv")

    rv_observation = tfd.MixtureSameFamily(
        mixture_distribution=rv_assignments,
        components_distribution=tfd.Exponential(sample_rates)
    )

    return (
        rv_prob.log_prob(sample_prob_1) +
        rv_prob.log_prob(sample_prob_2) +
        tf.reduce_sum(rv_rates.log_prob(sample_rates)) +
        tf.reduce_sum(rv_observation.log_prob(data))
    )


# MCMC
is_prior_gamma = True

number_of_steps = 25000
burnin = 2500

# set the chain's initial state
initial_chain_state = [
    tf.constant(0.5, name="init_prob"),
    tf.constant([dataset_mean_, dataset_mean_], name="init_rates")
]

unconstraining_bijectors = [
    tfp.bijectors.Identity(),
    tfp.bijectors.Identity()
]

# define closure over our joint_log_prob
if is_prior_gamma:
    # unnormalized_posterior_log_prob = lambda *args: joint_log_prob_with_gamma_priors(dataset, *args)
    unnormalized_posterior_log_prob = lambda *args: joint_log_prob_dirichlet(dataset, *args)
else:
    unnormalized_posterior_log_prob = lambda *args: joint_log_prob_with_uniform_priors(dataset, *args)

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

start_time = time.time()
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

print(f"MCMC took {(time.time() - start_time)/60:4.2f}m.")

new_step_size_initializer_ = kernel_results_.inner_results.is_accepted.mean()
print("acceptance rate: {}".format(new_step_size_initializer_))
print("final step size: {}".format(kernel_results_.inner_results.extra.step_size_assign[-100:].mean()))

# Plotting
lw = 1
plt.subplot(211)
plt.plot(posterior_rates_[:, 0], lw=lw, c='red',
         label=f"trace of rate 0. Mean: {np.mean(posterior_rates_[:, 0])}, SD: {np.std(posterior_rates_[:, 0])}")
plt.plot(posterior_rates_[:, 1], lw=lw, c='blue',
         label=f"trace of rate 1. Mean: {np.mean(posterior_rates_[:, 1])}, SD: {np.std(posterior_rates_[:, 1])}")
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
plt.hist(posterior_rates_[:, 0], color='red', bins=50, histtype="stepfilled")

plt.subplot(212)
plt.title(f"Posterior of rate of 1")
plt.hist(posterior_rates_[:, 1], color='blue', bins=50, histtype="stepfilled")

plt.tight_layout()
plt.show()

print(f"Rate 0. Mean: {np.mean(posterior_rates_[:, 0])}, SD: {np.std(posterior_rates_[:, 0])}")
print(f"Rate 1. Mean: {np.mean(posterior_rates_[:, 1])}, SD: {np.std(posterior_rates_[:, 1])}")
print(f"$p$: frequency of assignment to cluster 0. Mean: {np.mean(posterior_prob_)}")
print(posterior_prob_)


# Clustering
dist_1 = tfd.Exponential(rate=1. / np.mean(posterior_rates_[:, 0]))
dist_2 = tfd.Exponential(rate=1. / np.mean(posterior_rates_[:, 1]))

prob_assignment_1 = dist_1.prob(tf.cast(dataset, dtype='float64'))
prob_assignment_2 = dist_2.prob(tf.cast(dataset, dtype='float64'))

probs_assignments = tf.subtract(tf.cast(1., dtype='float64'), tf.div(prob_assignment_2,
                                                                     tf.add_n([prob_assignment_1, prob_assignment_2])))
# auc = tf.metrics.auc(labels, probs_assignments)
with tf.Session() as sess:
    sess.run(tf.initialize_local_variables())
    labels_, probs_assignments_ = sess.run([labels, probs_assignments])


fpr, tpr, thresholds = metrics.roc_curve(labels_, probs_assignments_)
auc = metrics.auc(fpr, tpr)

if auc < 0.5:
    fpr, tpr, thresholds = metrics.roc_curve(labels_, probs_assignments_, pos_label=0)
    auc = metrics.auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Exponential Mixture Model ROC')
plt.legend(loc="lower right")
plt.show()
