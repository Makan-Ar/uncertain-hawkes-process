import tensorflow as tf

tf.enable_eager_execution()

i = tf.constant(0)
while i < 1000:
    i = tf.add(i, 1)

print(i)

x = tf.random_uniform([2, 2])

for i in range(x.shape[0]):
  for j in range(x.shape[1]):
    print(x[i, j])

