import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

# build graph
rv_coin_flip_data = tfp.distributions.Bernoulli(probs=0.5, dtype=tf.int32)
num_trials = tf.constant([0, 1, 2, 3, 4, 5, 8, 15, 50, 500, 1000, 2000])

coin_flip_data = rv_coin_flip_data.sample(num_trials[-1])

# prepend a 0 onto tally of heads and tails, for zeroth flip
coin_flip_data = tf.pad(coin_flip_data, tf.constant([[1, 0]]), mode="CONSTANT")


# compute cumulative headcounts from 0 to 2000 flips, and then grab them at each of num_trials intervals
cumulative_headcounts = tf.gather(tf.cumsum(coin_flip_data), num_trials)


rv_observed_heads = tfp.distributions.Beta(concentration1=tf.to_float(1 + cumulative_headcounts),
                                           concentration0=tf.to_float(1 + num_trials - cumulative_headcounts))

probs_of_heads = tf.linspace(start=0., stop=1., num=100, name="linspace")


observed_probs_heads = tf.transpose(rv_observed_heads.prob(probs_of_heads[:, tf.newaxis]))

# Launch the graph in a session.
sess = tf.Session()


def evaluate(tensors):
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


[num_trials_, probs_of_heads_, observed_probs_heads_, cumulative_headcounts_] = \
    evaluate([num_trials, probs_of_heads, observed_probs_heads, cumulative_headcounts])

print(num_trials_)
print(probs_of_heads_)
print(observed_probs_heads_)
print(cumulative_headcounts_)
# For the already prepared, I'm using Binomial's conj. prior.
# plt.figure((16, 9))
for i in range(len(num_trials_)):
    sx = plt.subplot(len(num_trials_)/2, 2, i+1)
    plt.xlabel("$p$, probability of heads") \
    if i in [0, len(num_trials_)-1] else None
    plt.setp(sx.get_yticklabels(), visible=False)
    plt.plot(probs_of_heads_, observed_probs_heads_[i],
             label="observe %d tosses,\n %d heads" % (num_trials_[i], cumulative_headcounts_[i]))
    plt.fill_between(probs_of_heads_, 0, observed_probs_heads_[i],
                     color="blue", alpha=0.4)
    plt.vlines(0.5, 0, 4, color="k", linestyles="--", lw=1)
    leg = plt.legend()
    leg.get_frame().set_alpha(0.4)
    plt.autoscale(tight=True)


plt.suptitle("Bayesian updating of posterior probabilities", y=1.02,
             fontsize=14)
plt.tight_layout()
# plt.show()