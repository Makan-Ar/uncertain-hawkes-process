import tensorflow as tf
import tensorflow_probability as tfp

# tf.enable_eager_execution()

tfd = tfp.distributions

# lamda = tfd.Exponential(rate=1., name="lamda").sample()
# poisson_dist = tfd.Poisson(lamda, name="data_generator")
# poisson_sample = poisson_dist.sample()
#
# print(lamda)
# print(poisson_sample)
#
# writer = tf.summary.FileWriter('./graph-files/myg.g', tf.get_default_graph())
# with tf.Session() as sess:
#     print(sess.run([lamda, poisson_sample]))
# writer.close()

##################################################################

lambda_1 = tfd.Exponential(rate=1., name="lambda_1")  # stochastic variable
lambda_2 = tfd.Exponential(rate=1., name="lambda_2")  # stochastic variable
tau = tfd.Uniform(name="tau", low=0., high=10.)  # stochastic variable

p = tf.add(lambda_1.sample(), lambda_2.sample())

# deterministic variable since we are getting results of lambda's after sampling
new_deterministic_variable = tfd.Deterministic(name="deterministic_variable",  loc=p).sample()
new_deterministic_variables = tfd.Deterministic(name="deterministic_variable",  loc=p).sample()

with tf.Session() as sess:
    print(sess.run([p, new_deterministic_variable, new_deterministic_variables]))
