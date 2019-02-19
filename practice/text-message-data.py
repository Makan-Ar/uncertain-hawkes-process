import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

# EXPLORING THE DATA

# Defining our Data and assumptions
count_data = tf.constant([
    13, 24, 8, 24, 7, 35, 14, 11, 15, 11, 22, 22, 11, 57,
    11, 19, 29, 6, 19, 12, 22, 12, 18, 72, 32, 9, 7, 13,
    19, 23, 27, 20, 6, 17, 13, 10, 14, 6, 16, 15, 7, 2,
    15, 15, 19, 70, 49, 7, 53, 22, 21, 31, 19, 11, 18, 20,
    12, 35, 17, 23, 17, 4, 2, 31, 30, 13, 27, 0, 39, 37,
    5, 14, 13, 22,
], dtype=tf.float32)
n_count_data = tf.size(count_data)
days = tf.range(n_count_data)


def evaluate(tensors, sess):
    """Evaluates Tensor or EagerTensor to Numpy `ndarray`s.
    Args:
    tensors: Object of `Tensor` or EagerTensor`s; can be `list`, `tuple`,
      `namedtuple` or combinations thereof.

    Returns:
      ndarrays: Object with same structure as `tensors` except with `Tensor` or
        `EagerTensor`s replaced by Numpy `ndarray`s.
    """
    if tf.executing_eagerly():
        return tf.contrib.framework.nest.pack_sequence_as(tensors,
                                                          [t.numpy() if tf.contrib.framework.is_tensor(t) else t
                                                           for t in tf.contrib.framework.nest.flatten(tensors)])
    return sess.run(tensors)

# Convert from TF to numpy.

# [
#     count_data_,
#     n_count_data_,
#     days_,
# ] = evaluate([
#     count_data,
#     n_count_data,
#     days,
# ])
#
# # Visualizing the Results
#
# plt.figure(figsize=(12.5, 4))
# plt.bar(days_, count_data_, color="#5DA5DA")
# plt.xlabel("Time (days)")
# plt.ylabel("count of text-msgs received")
# plt.title("Did the user's texting habits change over time?")
# plt.xlim(0, n_count_data_)
# plt.show()

##########################################################
# DEFINING THE JOINT PROBABILITY


def joint_log_prob(count_data, lambda_1, lambda_2, tau):
    tfd = tfp.distributions

    alpha = (1. / tf.reduce_mean(count_data))
    rv_lambda_1 = tfd.Exponential(rate=alpha)
    rv_lambda_2 = tfd.Exponential(rate=alpha)

    rv_tau = tfd.Uniform()

    lambda_ = tf.gather([lambda_1, lambda_2],
                        indices=tf.to_int32(
                            tau * tf.to_float(tf.size(count_data)) <= tf.to_float(tf.range(tf.size(count_data)))))
    rv_observation = tfd.Poisson(rate=lambda_)

    #  once we've specified the probabilistic model, we return the sum of the log_probs
    return (rv_lambda_1.log_prob(lambda_1)
            + rv_lambda_2.log_prob(lambda_2)
            + rv_tau.log_prob(tau)
            + tf.reduce_sum(rv_observation.log_prob(count_data)))

##########################################################
# MCMC

# Set the chain's start state.
initial_chain_state = [
    tf.to_float(tf.reduce_mean(count_data)) * tf.ones([], dtype=tf.float32, name="init_lambda1"),
    tf.to_float(tf.reduce_mean(count_data)) * tf.ones([], dtype=tf.float32, name="init_lambda2"),
    0.5 * tf.ones([], dtype=tf.float32, name="init_tau"),
]

# Since HMC operates over unconstrained space, we need to transform the
# samples so they live in real-space.
unconstraining_bijectors = [
    tfp.bijectors.Exp(),  # Maps a positive real to R.
    tfp.bijectors.Exp(),  # Maps a positive real to R.
    tfp.bijectors.Sigmoid(),  # Maps [0,1] to R.
]


# Define a closure over our joint_log_prob.
def unnormalized_log_posterior(lambda1, lambda2, tau):
    return joint_log_prob(count_data, lambda1, lambda2, tau)


# Initialize the step_size. (It will be automatically adapted.)
with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    step_size = tf.get_variable(
        name='step_size',
        initializer=tf.constant(0.05, dtype=tf.float32),
        trainable=False,
        use_resource=True
    )

# Sample from the chain.
[
    lambda_1_samples,
    lambda_2_samples,
    posterior_tau,
], kernel_results = tfp.mcmc.sample_chain(
    num_results=1000,
    num_burnin_steps=100,
    current_state=initial_chain_state,
    kernel=tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=unnormalized_log_posterior,
            num_leapfrog_steps=2,
            step_size=step_size,
            step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(),
            state_gradients_are_stopped=True),
        bijector=unconstraining_bijectors))

tau_samples = tf.floor(posterior_tau * tf.to_float(tf.size(count_data)))

# tau_samples, lambda_1_samples, lambda_2_samples contain
# N samples from the corresponding posterior distribution
N = tf.shape(tau_samples)[0]
expected_texts_per_day = tf.zeros(n_count_data)

# Initialize any created variables.
init_g = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()

##########################################################
# EVALUATING THE GRAPH

# Launch the graph in a session.
sess = tf.Session()

evaluate(init_g, sess)
evaluate(init_l, sess)
[
    lambda_1_samples_,
    lambda_2_samples_,
    tau_samples_,
    kernel_results_,
    N_,
    expected_texts_per_day_,
    count_data_,
    n_count_data_,
] = evaluate([
    lambda_1_samples,
    lambda_2_samples,
    tau_samples,
    kernel_results,
    N,
    expected_texts_per_day,
    count_data,
    n_count_data,
], sess)

print("acceptance rate: {}".format(
    kernel_results_.inner_results.is_accepted.mean()))
print("final step size: {}".format(
    kernel_results_.inner_results.extra.step_size_assign[-100:].mean()))

##########################################################
# PLOTTING POSTERIOR SAMPLES


plt.figure(figsize=(12.5, 15))
# histogram of the samples:

ax = plt.subplot(311)
ax.set_autoscaley_on(False)

plt.hist(lambda_1_samples_, histtype='stepfilled', bins=30, alpha=0.85,
         label=r"posterior of $\lambda_1$", color='red', density=True)
plt.legend(loc="upper left")
plt.title(r"""Posterior distributions of the variables $\lambda_1,\;\lambda_2,\;\tau$""")
plt.xlim([15, 30])
plt.xlabel(r"$\lambda_1$ value")

ax = plt.subplot(312)
ax.set_autoscaley_on(False)
plt.hist(lambda_2_samples_, histtype='stepfilled', bins=30, alpha=0.85,
         label=r"posterior of $\lambda_2$", color='purple', density=True)
plt.legend(loc="upper left")
plt.xlim([15, 30])
plt.xlabel(r"$\lambda_2$ value")

plt.subplot(313)
w = 1.0 / tau_samples_.shape[0] * np.ones_like(tau_samples_)
plt.hist(tau_samples_, bins=n_count_data_, alpha=1,
         label=r"posterior of $\tau$",
         color='green', weights=w, rwidth=2.)
plt.xticks(np.arange(n_count_data_))

plt.legend(loc="upper left")
plt.ylim([0, .75])
plt.xlim([35, len(count_data_)-20])
plt.xlabel(r"$\tau$ (in days)")
plt.ylabel(r"probability")
plt.show()


#################################################################
# what is the expected number of texts per day

plt.figure(figsize=(12.5, 9))

for day in range(0, n_count_data_):
    # ix is a bool index of all tau samples corresponding to
    # the switchpoint occurring prior to value of 'day'
    ix = day < tau_samples_
    # Each posterior sample corresponds to a value for tau.
    # for each day, that value of tau indicates whether we're "before"
    # (in the lambda1 "regime") or
    #  "after" (in the lambda2 "regime") the switchpoint.
    # by taking the posterior sample of lambda1/2 accordingly, we can average
    # over all samples to get an expected value for lambda on that day.
    # As explained, the "message count" random variable is Poisson distributed,
    # and therefore lambda (the poisson parameter) is the expected value of
    # "message count".
    expected_texts_per_day_[day] = (lambda_1_samples_[ix].sum() + lambda_2_samples_[~ix].sum()) / N_


plt.plot(range(n_count_data_), expected_texts_per_day_, lw=4, color="#E24A33",
         label="expected number of text-messages received")
plt.xlim(0, n_count_data_)
plt.xlabel("Day")
plt.ylabel("Expected # text-messages")
plt.title("Expected number of text-messages received")
plt.ylim(0, 60)
plt.bar(np.arange(len(count_data_)), count_data_, color="#5DA5DA", alpha=0.65,
        label="observed texts per day")

plt.legend(loc="upper left")
plt.show()