import tensorflow as tf

x = tf.Variable(2.0, name = 'x')
y = x * x
optimizer = tf.train.GradientDescentOptimizer(0.03)
grads_and_vars = optimizer.compute_gradients(y)

print x.name
print grads_and_vars[0][1].name



init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
