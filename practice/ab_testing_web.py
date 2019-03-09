import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

tfd = tfp.distributions

p = tfd.Uniform(low=0., high=0., name='p')

p_true = tf.constant(0.05, name="p_true")
num_customers = tf.constant(1500, dtype=tf.int32, name="num_customers")

occurrences = tfd.Bernoulli(probs=p_true).sample(num_customers, seed=6.45)

# with tf.Session() as sess:
#     print(sess.run([x, tf.reduce_mean(tf.to_float(x)), tf.reduce_sum(x)]))


def joint_log_probability(data, p):
    rv_prob_p = tfd.Uniform(low=0., high=0.)
    rv_x = tfd.Bernoulli(probs=p)

    # log of the prior * the likelihood
    return (
        rv_prob_p.log_prob(p) + tf.reduce_sum(rv_x.log_prob(data))
    )


number_of_steps = 48000  # @param {type:"slider", min:2000, max:50000, step:100}
burnin = 25000 # @param {type:"slider", min:0, max:30000, step:100}
leapfrog_steps = 2  # @param {type:"slider", min:1, max:9, step:1}

# Set the chain's start state.
initial_chain_state = [
    tf.reduce_mean(tf.to_float(occurrences))
    * tf.ones([], dtype=tf.float32, name="init_prob_A")
]

# Since HMC operates over unconstrained space, we need to transform the
# samples so they live in real-space.
unconstraining_bijectors = [
    tfp.bijectors.Identity()   # Maps R to R.
]

# Define a closure over our joint_log_prob.
# The closure makes it so the HMC doesn't try to change the `occurrences` but
# instead determines the distributions of other parameters that might generate
# the `occurrences` we observed.
unnormalized_posterior_log_prob = lambda *args: joint_log_probability(occurrences, *args)

# Initialize the step_size. (It will be automatically adapted.)
with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    step_size = tf.get_variable(
        name='step_size',
        initializer=tf.constant(0.5, dtype=tf.float32),
        trainable=False,
        use_resource=True
    )

# Defining the HMC
hmc = tfp.mcmc.TransformedTransitionKernel(
    inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=unnormalized_posterior_log_prob,
        num_leapfrog_steps=leapfrog_steps,
        step_size=step_size,
        step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(num_adaptation_steps=burnin),
        state_gradients_are_stopped=True),
    bijector=unconstraining_bijectors)

# Sampling from the chain.
[
    posterior_prob_A
], kernel_results = tfp.mcmc.sample_chain(
    num_results=number_of_steps,
    num_burnin_steps=burnin,
    current_state=initial_chain_state,
    kernel=hmc)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    posterior_prob_A_, kernel_results_ = sess.run([posterior_prob_A, kernel_results])

print("acceptance rate: {}".format(
    kernel_results_.inner_results.is_accepted.mean()))

burned_prob_A_trace_ = posterior_prob_A_[burnin:]
plt.title("Posterior distribution of $p_A$, the true effectiveness of site A")
plt.vlines(0.05, 0, 90, linestyle="--", label="true $p_A$ (unknown)")
plt.hist(burned_prob_A_trace_, bins=25, histtype="stepfilled", normed=True)
plt.legend()
plt.show()
