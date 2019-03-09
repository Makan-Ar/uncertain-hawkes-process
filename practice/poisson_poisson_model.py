import tensorflow as tf
import tensorflow_probability as tfp
import poisson_uncertain_simulator as pus

tfd = tfp.distributions

data = pus.UncertainPoissonModel(lambda_1=0.4, exp_beta_1=0.7, lambda_2=0.7, exp_beta_2=0.5)
# data.plot_poisson_1()
# data.plot_poisson_2()
# data.plot_poisson_uncertain()
print(data.poisson_1.timestamps)

# lambda_1 = tf.constant(2., name="lambda_1")
# lambda_2 = tf.constant(6., name="lambda_1")
#
# num_sample_1 = tf.constant(500, name="num_sample_1")
# num_sample_2 = tf.constant(650, name="num_sample_2")
#
# data_generator_1 = tfd.Poisson(rate=lambda_1, name="poisson_generator_1")
# data_generator_2 = tfd.Poisson(rate=lambda_2, name="poisson_generator_2")
#
# sample_1 = data_generator_1.sample(10)
# sample_2 = data_generator_2.sample(50)
#
# with tf.Session() as sess:
#     print(sess.run([sample_1, sample_2]))
