import tensorflow as tf


###################################################
# # Create and visualize a simple graph by tensorboard
#
# a = tf.constant(2, name="a")
# b = tf.constant(3, name="b")
# x = tf.add(a, b)
#
# writer = tf.summary.FileWriter('./graph-files/myg.g', tf.get_default_graph())
# with tf.Session() as sess:
#     # writer = tf.summary.FileWriter('./graphs', sess.graph)
#     print(sess.run(x))
# writer.close()
#
# # then run: tensorboard --logdir="./graph-files/myg.g" --port 8000

###################################################
# # Create constant tensors
#
# a = tf.constant([2, 3], name='a')
# b = tf.constant([[1, 9], [2, 3]], name='b')
# x = tf.multiply(a, b, name='mul')
#
# with tf.Session() as sess:
#     print(sess.run(x))
#
# # ones_like = tf.ones_like(a, name="ones-like")
# zeros = tf.zeros([10, 4], dtype=tf.float32, name="zeros")
# zero_like = tf.zeros_like(b, name="zero-like")
# p = tf.fill([9, 3], 9, name="9-tensor")
#
# tf_lin = tf.lin_space(10.0, 13.0, 4)  # ==> [10. 11. 12. 13.]
# tf_range = tf.range(3, 18, 3)  # ==> [3 6 9 12 15]
# # keep in mind that tensor objects are not iterable
#
# writer = tf.summary.FileWriter('./graph-files/myg.g', tf.get_default_graph())
#
# with tf.Session() as sess:
#     print(sess.run([zeros, zero_like, p]))

###################################################
# Create variable tensors

# create variables with tf.Variable
s = tf.Variable(2, name="scalar")
m = tf.Variable([[0, 1], [2, 3]], name="matrix")
W = tf.Variable(tf.zeros([784, 10]))

# create variables with tf.get_variable
s = tf.get_variable("scalar", initializer=tf.constant(2))
m = tf.get_variable("matrix", initializer=tf.constant([[0, 1], [2, 3]]))
W = tf.get_variable("big_matrix", shape=(700, 100), initializer=tf.zeros_initializer())

# x.initializer # init op
# x.value() # read op
# x.assign(...) # write op
# x.assign_add(...) # and more

# with tf.Session() as sess:
# 	print(sess.run(W))   >> FailedPreconditionError: Attempting to use uninitialized value Variable

# You must initialize your variables

# The easiest way is initializing all variables at once:
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

# Initialize only a subset of variables:
with tf.Session() as sess:
    sess.run(tf.variables_initializer([s, m]))

# Initialize a single variable
with tf.Session() as sess:
    sess.run(W.initializer)

# W is a random 700 x 100 variable object
with tf.Session() as sess:
    sess.run(W.initializer)
    print(W)  # >> Tensor("Variable/read:0", shape=(700, 100), dtype=float32)
    print(W.eval())	 # Similar to print(sess.run(W)) will give the numpy format

# Assigning a value
W = tf.Variable(10)
W.assign(100)
with tf.Session() as sess:
    sess.run(W.initializer)
    print(W.eval())  # >> 10 why? b/c W.assign creates an assign operator which must be run in the session

W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
    sess.run(W.initializer)
    sess.run(assign_op)
    print(W.eval())  # >> 100 Yay!

# You don’t need to initialize variable because assign_op does it for you. In fact, initializer op is the assign op that
# assigns the variable’s initial value to the variable itself.

# create a variable whose original value is 2
my_var = tf.Variable(2, name="my_var")

# assign a * 2 to a and call that op a_times_two
my_var_times_two = my_var.assign(2 * my_var)

with tf.Session() as sess:
    sess.run(my_var.initializer)
    sess.run(my_var_times_two)  # >> the value of my_var now is 4
    sess.run(my_var_times_two)  # >> the value of my_var now is 8
    sess.run(my_var_times_two)  # >> the value of my_var now is 16

my_var = tf.Variable(10)

with tf.Session() as sess:
    sess.run(my_var.initializer)

    # increment by 10
    sess.run(my_var.assign_add(10))  # >> 20
    # decrement by 2
    sess.run(my_var.assign_sub(2))  # >> 18

    # assign_add and assign_sub do not init the variable itself!


# each session maintains its own copy of variable
W = tf.Variable(10)

sess1 = tf.Session()
sess2 = tf.Session()

sess1.run(W.initializer)
sess2.run(W.initializer)

print(sess1.run(W.assign_add(10)))  # >> 20
print(sess2.run(W.assign_sub(2)))  # >> 8

print(sess1.run(W.assign_add(100)))  # >> 120
print(sess2.run(W.assign_sub(50)))  # >> -42

sess1.close()
sess2.close()

# Sometimes, we will have two more two independent ops but you’d like to specify which op should be run first, then you
# use tf.Graph.control_dependencies(control_inputs)

# your graph g have 5 ops: a, b, c, d, e
# g = tf.get_default_graph()
# with g.control_dependencies([a, b, c]):
    # 'd' and 'e' will only run after 'a', 'b', and 'c' have executed.
    # d = ...
    # e = …


######################################################
# PLACEHOLDERS

# create a placeholder for a vector of 3 elements, type tf.float32
a = tf.placeholder(tf.float32, shape=[3])

b = tf.constant([5, 5, 5], tf.float32)

# use the placeholder as you would a constant or a variable
c = a + b  # short for tf.add(a, b)

with tf.Session() as sess:
    print(sess.run(c, feed_dict={a: [1, 2, 3]})) 	# the tensor a is the key, not the string ‘a’ # >> [6, 7, 8]

# feeding in multiple points
with tf.Session() as sess:
    for a_value in []:
        print(sess.run(c, {a: a_value}))


# tf.Graph.is_feedable(tensor) #  True if and only if tensor is feedable.
