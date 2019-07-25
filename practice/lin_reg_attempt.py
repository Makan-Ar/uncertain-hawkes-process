import time
import likelihood_utils
import tensorflow as tf

DATA_FILE = "birth_life_2010.txt"

# # Step 1: read in data from the .txt file
# # data is a numpy array of shape (190, 2), each row is a data point
# data, n_samples = utils.read_birth_life_data(DATA_FILE)
#
# X = tf.placeholder(dtype=tf.float32, name='X')
# Y = tf.placeholder(dtype=tf.float32, name='Y')
#
# w = tf.get_variable("weights", initializer=tf.constant(0.0))
# b = tf.get_variable("bias", initializer=tf.constant(0.0))
#
#
# Y_predicted = b + w * X
#
# loss = tf.square(Y - Y_predicted, name='loss')
#
# optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#
# writer = tf.summary.FileWriter('./graph-files/myg.g', tf.get_default_graph())
#
# with tf.Session() as sess:
#     start_time = time.time()
#     sess.run(tf.global_variables_initializer())
#     for i in range(100):  # run for 100 epochs
#         for x, y in data:
#             sess.run(optimizer, feed_dict={X: x, Y: y})
#
#     # Step 9: output the values of w and b
#     w_out, b_out = sess.run([w, b])
#
#     print(b_out)
#     print(w_out)
#     print('Total time: {0} seconds'.format(time.time() - start_time))
#
# writer.close()

# ###################################################################
n_epoch = 100
batch_size = 20

# With proper data loading

# step 1: load the data with numpy or something
data, n_samples = likelihood_utils.read_birth_life_data(DATA_FILE)

# step 2: create a tf dataset and manipulate it as needed
dataset = tf.data.Dataset.from_tensor_slices((data[:, 0], data[:, 1]))  # tuple shape matches the iterator output
# dataset = dataset.batch(batch_size)
# dataset = dataset.shuffle(1000)
# dataset = dataset.repeat(100)
# dataset = dataset.map(lambda x: tf.one_hot(x, 10))

# step 3: create an iterator over it, lastly, run the session over it
iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
X, Y = iterator.get_next()
init_train = iterator.make_initializer(dataset)

w = tf.get_variable("weights", initializer=tf.constant(0.0))
b = tf.get_variable("bias", initializer=tf.constant(0.0))


Y_predicted = b + w * X

loss = tf.square(Y - Y_predicted, name='loss')

optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

writer = tf.summary.FileWriter('./graph-files/myg.g', tf.get_default_graph())

with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    for i in range(n_epoch):  # run for 100 epochs
        sess.run(init_train)

        # the only way I know to catch the end of the iterator
        try:
            while True:
                sess.run(optimizer)
        except tf.errors.OutOfRangeError:
            pass

    # Step 9: output the values of w and b
    w_out, b_out = sess.run([w, b])

    print(b_out)
    print(w_out)
    print('Total time: {0} seconds'.format(time.time() - start_time))

writer.close()
