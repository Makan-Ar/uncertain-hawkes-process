import time
import numpy as np
import tensorflow as tf
# import tensorflow_probability as tfp
# import tensorflow_probability.python.edward2 as ed

# beta = ed.Beta(concentration1=1, concentration2=1)

# normal_dist = tfp.distributions.Normal(0, 1)
# x = normal_dist.sample()
# print(x)
# print(normal_dist.prob(x))

# Following tut: https://www.tensorflow.org/tutorials/eager/eager_basics
# This enables a more interactive frontend
tf.enable_eager_execution()

# print(tf.add(1, 2))
#
#
# nparr = np.ones((3, 3))
#
# tensor = tf.multiply(nparr, 93)
# print(tensor)
# print(tensor.numpy())
#
# x = tf.random_uniform([3, 3])
# print("Is there a GPU available: "),
# print(tf.test.is_gpu_available())
#
# print("Is the Tensor on GPU #0:  "),
# print(x.device.endswith('GPU:0'))
#
# print(tensor.device)


def time_matmul(x):
    start = time.time()
    for loop in range(10):
        tf.matmul(x, x)

    result = time.time() - start

    print("10 loops: {:0.2f}ms".format(1000 * result))


# Force execution on CPU
print("On CPU:")
with tf.device("CPU:0"):
    start = time.time()
    x = tf.random_uniform([1000, 1000])
    assert x.device.endswith("CPU:0")
    time_matmul(x)
    print("Uniform GPU: {:0.2f}ms".format(1000 * (time.time() - start)))

# Force execution on GPU #0 if available
if tf.test.is_gpu_available():
    with tf.device("GPU:0"):  # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
        start = time.time()
        x = tf.random_uniform([1000, 1000])
        assert x.device.endswith("GPU:0")
        time_matmul(x)

        print("Uniform CPU: {:0.2f}ms".format(1000 * (time.time() - start)))
