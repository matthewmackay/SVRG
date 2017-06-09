#########################
### TODO: 
### Add code to compute full gradient (see Goodfellow's example online)
### Add in variable scopes
### Add in savers
### Add in Tensorboard
### Put variables into dictionary
### Add functions for computing cost, updates (if possible?)
### Add in learning rate scheduling to NN
### Figure out why SVRG isn't promoting faster learning in NN
### Implement hyperparameters as flags
### Put in adaptive learning rate
### Learn about this get_variable stuff
### Make NN and log reg instances of same super class
#########################

from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX[:10000,:]
trY = trY[:10000,:]


n = trX.shape[0]

def sgd_log(num_updates, track_every):
	### Hyperparameters
	learning_rate = 0.005
	training_epochs = 25
	gamma = 1e-4

	### tf Graph Input
	x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
	y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

	### Set model weights
	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))

	### Construct model
	pred = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax

	### Define loss function
	cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices = 1)) + gamma * tf.reduce_sum(tf.square(W))
	[dW, db] = tf.gradients(cost, [W, b])

	### Formulate and apply SGD update
	update_W = learning_rate * dW
	update_b = learning_rate * db
	adj_W = W.assign(W - update_W)
	adj_b = b.assign(b - update_b)

	init = tf.global_variables_initializer()

	### Train our model, keeping track of cost & variance of updates throughout training
	costs = []
	variances = []
	with tf.Session() as sess:
		sess.run(init)

		### Run SGD for specified number of updates
		for i in range(num_updates):
			idx = np.random.randint(low = 0, high = n)
			sess.run([adj_b, adj_W], feed_dict = {x: trX[idx, None, :], y: trY[idx, None, :]})

			### Print out cost/variance when specified
			if i % track_every == 0:
				### Retrieve cost
				curr_cost, gt_update = sess.run([cost, update_W], feed_dict = {x: trX, y: trY})
				costs.append(curr_cost)

				### Estimate variance of our update
				curr_variance = 0
				for j in range(n):
					#idx = np.random.randint(low = 0, high = n)
					curr_update = update_W.eval(feed_dict = {x: trX[j, None, :], y: trY[j, None, :]})
					curr_variance += np.sum(np.square(curr_update - gt_update))
				variances.append(curr_variance)
				
				### Print our metrics
				print("Iteration: {0}\tCost: {1}\tVariance: {2}".format(i, curr_cost, curr_variance))
		
	return costs, variances

def svrg_log(num_updates, track_every):
	### Hyperparameters
	m = 100 #How many iterations until recalculate batch gradient
	learning_rate = 0.025 	
	gamma = 1e-4	#Strength of L2 regularization
	num_iterations = 10 #For total number of iterations, multiply by m

	### tf Graph Input
	x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
	y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

	### Placeholder dictating whether SVRG or SGD update should be applied
	svrg_flag = tf.placeholder(tf.bool)

	### Set model weights
	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))

	### Variables to be used for our control variate
	control_W = tf.Variable(tf.zeros([784, 10]))
	control_b = tf.Variable(tf.zeros([10]))
	batch_dcontrol_W = tf.Variable(tf.zeros([784, 10]))
	batch_dcontrol_b = tf.Variable(tf.zeros([10]))

	### Construct model
	pred = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax
	control_pred = tf.nn.softmax(tf.matmul(x, control_W) + control_b)

	### Define loss function
	cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices = 1)) + gamma * tf.reduce_sum(tf.square(W)) #Cost for actual weights
	control_cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(control_pred), reduction_indices = 1)) + gamma * tf.reduce_sum(tf.square(control_W)) #Cost for control weights
	
	### Get gradients
	[dW, db] = tf.gradients(cost, [W, b])
	[dcontrol_W, dcontrol_b] = tf.gradients(control_cost, [control_W, control_b])

	### Form SVRG and SGD updates
	update_W = tf.cond(svrg_flag, lambda: learning_rate * (dW - dcontrol_W + batch_dcontrol_W), lambda: learning_rate * dW)
	update_b = tf.cond(svrg_flag, lambda: learning_rate * (db - dcontrol_b + batch_dcontrol_b), lambda: learning_rate * db)
	
	### Adjust parameters based on computed update
	adj_W = W.assign(W - update_W)
	adj_b = b.assign(b - update_b)

	### Create nodes for updating our control variate (done every m iterations)
	reset_W = control_W.assign(W)
	reset_b = control_b.assign(b)
	reset_batch_db = batch_dcontrol_b.assign(db)
	reset_batch_dW = batch_dcontrol_W.assign(dW)

	### Train our model
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)

		### We run one iteration of SGD to initialize the weights for SVRG
		idx = np.random.randint(low = 0, high = n)
		sess.run([adj_W, adj_b], feed_dict = {x: trX[idx, None, :], y: trY[idx, None, :], svrg_flag: False})
		sess.run([reset_W, reset_b, reset_batch_db, reset_batch_dW], feed_dict = {x: trX, y: trY})

		costs = []
		variances = []
		i = 0
		while (i < num_updates):
			t = 0
			while (t < m and i < num_updates):
				### Apply control variate update
				idx = np.random.randint(low = 0, high = n)
				sess.run([adj_b, adj_W], feed_dict = {x: trX[idx, None, :], y: trY[idx, None, :], svrg_flag: True})
				
				###Track cost/variance when specified
				if (i % track_every == 0):
					curr_cost, gt_update = sess.run([cost, update_W], feed_dict = {x: trX, y: trY, svrg_flag: True})
					costs.append(curr_cost)
					
					### Estimate variance of our update
					curr_variance = 0
					for j in range(n):
						#idx = np.random.randint(low = 0, high = n)
						curr_update = update_W.eval(feed_dict = {x: trX[j, None, :], y: trY[j, None, :], svrg_flag: True})
						curr_variance += np.sum(np.square(curr_update - gt_update))
					variances.append(curr_variance) 

					print("Iteration: {0}\tCost: {1}\tVariance: {2}".format(i, curr_cost, curr_variance))

				### Increment our counters	
				t += 1
				i += 1

			### Update our control values every m iterations
			sess.run([reset_W, reset_b, reset_batch_db, reset_batch_dW], feed_dict = {x: trX, y: trY})

	return costs, variances

def sgd_nn(num_updates, track_every, learning_rate, batch_size):
	### Hyperparameters
	gamma = 1e-4

	### tf Graph Input
	x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
	y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

	### Set model weights
	Wh = tf.Variable(tf.random_normal([784, 100], stddev = 0.01))
	bh = tf.Variable(tf.random_normal([100], stddev = 0.01))

	Wo = tf.Variable(tf.random_normal([100, 10], stddev = 0.01))
	bo = tf.Variable(tf.random_normal([10], stddev = 0.01))

	### Construct model
	h = tf.nn.sigmoid(tf.matmul(x, Wh) + bh)
	pred = tf.nn.softmax(tf.matmul(h, Wo) + bo)  # Softmax

	### Define loss function
	cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices = 1)) + gamma * (tf.reduce_sum(tf.square(Wh)) + tf.reduce_sum(tf.square(Wo)))
	[dWh, dbh, dWo, dbo] = tf.gradients(cost, [Wh, bh, Wo, bo])

	### Formulate and apply SGD update
	update_Wh = learning_rate * dWh
	update_bh = learning_rate * dbh
	update_Wo = learning_rate * dWo
	update_bo = learning_rate * dbo
	
	adj_Wh = Wh.assign(Wh - update_Wh)
	adj_bh = bh.assign(bh - update_bh)
	adj_Wo = Wo.assign(Wo - update_Wo)
	adj_bo = bo.assign(bo - update_bo)

	init = tf.global_variables_initializer()

	### Train our model, keeping track of cost & variance of updates throughout training
	costs = []
	variances = []
	with tf.Session() as sess:
		sess.run(init)

		### Run SGD for specified number of updates
		for i in range(num_updates):
        	### Apply next mini-batch update (we use batch size of 10)
			idx = (i * batch_size) % n
			sess.run([adj_bh, adj_Wh, adj_Wo, adj_bo], feed_dict = {x: trX[idx:idx+batch_size, :], y: trY[idx:idx+batch_size, :]})        	

			### Print out cost/variance when specified
			if i % track_every == 0:
				### Retrieve cost
				curr_cost, gt_update_Wh, gt_update_Wo = sess.run([cost, update_Wh, update_Wo], feed_dict = {x: trX, y: trY})
				costs.append(curr_cost)

				### Estimate variance of our update
				curr_variance = 0
				for j in range(100):
					curr_idx = (j * batch_size) % n
					curr_update_Wh, curr_update_Wo = sess.run([update_Wh, update_Wo], feed_dict = {x: trX[curr_idx:curr_idx+batch_size, :], y: trY[curr_idx:curr_idx+batch_size, :]})   
					curr_variance += np.sum(np.square(curr_update_Wh - gt_update_Wh)) 
					curr_variance += np.sum(np.square(curr_update_Wo - gt_update_Wo)) 
				variances.append(curr_variance)
				
				### Print our metrics
				print("Iteration: {0}\tCost: {1}\tVariance: {2}".format(i, curr_cost, curr_variance))
		
	return costs, variances


def svrg_nn(num_updates, track_every, learning_rate, batch_size):
	### Hyperparameters
	gamma = 1e-4
	m = 100

	### tf Graph Input
	x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
	y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

	### Placeholder dictating whether SVRG or SGD update should be applied
	svrg_flag = tf.placeholder(tf.bool)

	### Set model weights
	Wh = tf.Variable(tf.random_normal([784, 100], stddev = 0.01))
	bh = tf.Variable(tf.random_normal([100], stddev = 0.01))

	Wo = tf.Variable(tf.random_normal([100, 10], stddev = 0.01))
	bo = tf.Variable(tf.random_normal([10], stddev = 0.01))

	### Variables to be used for our control variate
	control_Wh = tf.Variable(tf.zeros([784, 100]))
	control_bh = tf.Variable(tf.zeros([100]))
	
	control_Wo = tf.Variable(tf.zeros([100, 10]))
	control_bo = tf.Variable(tf.zeros([10]))

	batch_dcontrol_Wh = tf.Variable(tf.zeros([784, 100]))
	batch_dcontrol_bh = tf.Variable(tf.zeros([100]))

	batch_dcontrol_Wo = tf.Variable(tf.zeros([100, 10]))
	batch_dcontrol_bo = tf.Variable(tf.zeros([10]))

	### Construct model
	h = tf.nn.sigmoid(tf.matmul(x, Wh) + bh)
	pred = tf.nn.softmax(tf.matmul(h, Wo) + bo)  # Softmax

	control_h = tf.nn.sigmoid(tf.matmul(x, control_Wh) + control_bh)
	control_pred = tf.nn.softmax(tf.matmul(control_h, control_Wo) + control_bo)

	### Define loss function
	cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices = 1)) + gamma * (tf.reduce_sum(tf.square(Wh)) + tf.reduce_sum(tf.square(Wo)))
	control_cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(control_pred), reduction_indices = 1)) + gamma * (tf.reduce_sum(tf.square(control_Wh)) 
		+ tf.reduce_sum(tf.square(control_Wo)))
	
	[dWh, dbh, dWo, dbo] = tf.gradients(cost, [Wh, bh, Wo, bo])
	[dcontrol_Wh, dcontrol_bh, dcontrol_Wo, dcontrol_bo] = tf.gradients(control_cost, [control_Wh, control_bh, control_Wo, control_bo])

	### Formulate and apply SGD update
	update_Wh = tf.cond(svrg_flag, lambda: learning_rate * (dWh - dcontrol_Wh + batch_dcontrol_Wh), lambda: learning_rate * dWh)
	update_bh = tf.cond(svrg_flag, lambda: learning_rate * (dbh - dcontrol_bh + batch_dcontrol_bh), lambda: learning_rate * dbh)
	update_Wo = tf.cond(svrg_flag, lambda: learning_rate * (dWo - dcontrol_Wo + batch_dcontrol_Wo), lambda: learning_rate * dWo)
	update_bo = tf.cond(svrg_flag, lambda: learning_rate * (dbo - dcontrol_bo + batch_dcontrol_bo), lambda: learning_rate * dbo)
	
	adj_Wh = Wh.assign(Wh - update_Wh)
	adj_bh = bh.assign(bh - update_bh)
	adj_Wo = Wo.assign(Wo - update_Wo)
	adj_bo = bo.assign(bo - update_bo)

	### Create nodes for updating our control variate (done every m iterations)
	reset_Wh = control_Wh.assign(Wh)
	reset_bh = control_bh.assign(bh)
	reset_Wo = control_Wo.assign(Wo)
	reset_bo = control_bo.assign(bo)

	reset_batch_dbh = batch_dcontrol_bh.assign(dbh)
	reset_batch_dWh = batch_dcontrol_Wh.assign(dWh)
	reset_batch_dbo = batch_dcontrol_bo.assign(dbo)
	reset_batch_dWo = batch_dcontrol_Wo.assign(dWo)


	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)

		### We run one iteration of SGD to initialize the weights for SVRG
		idx = np.random.randint(low = 0, high = n-batch_size)
		sess.run([adj_Wh, adj_bh, adj_Wo, adj_bo], feed_dict = {x: trX[idx:idx+batch_size, :], y: trY[idx:idx+batch_size, :], svrg_flag: False})
		sess.run([reset_Wh, reset_bh, reset_Wo, reset_bo, reset_batch_dbh, reset_batch_dWh, reset_batch_dbo, reset_batch_dWo], feed_dict = {x: trX, y: trY})

		costs = []
		variances = []
		i = 0
		while (i < num_updates):
			t = 0
			while (t < m and i < num_updates):
				### Apply control variate update
				idx = (i * batch_size) % n
				sess.run([adj_bh, adj_Wh, adj_Wo, adj_bo], feed_dict = {x: trX[idx:idx+batch_size, :], y: trY[idx:idx+batch_size, :], svrg_flag: True})
				
				###Track cost/variance when specified
				if (i % track_every == 0):
					curr_cost, gt_update_Wh, gt_update_Wo = sess.run([cost, update_Wh, update_Wo], feed_dict = {x: trX, y: trY, svrg_flag: True})
					costs.append(curr_cost)
					
					### Estimate variance of our update
					curr_variance = 0
					for j in range(100):
						curr_idx = (j * batch_size) % n
						curr_update_Wh, curr_update_Wo = sess.run([update_Wh, update_Wo], feed_dict = {x: trX[curr_idx:curr_idx+batch_size, :], y: trY[curr_idx:curr_idx+batch_size, :], svrg_flag: True})
						curr_variance += np.sum(np.square(curr_update_Wh - gt_update_Wh))
						curr_variance += np.sum(np.square(curr_update_Wo - gt_update_Wo))
					variances.append(curr_variance) 

					print("Iteration: {0}\tCost: {1}\tVariance: {2}".format(i, curr_cost, curr_variance))

				### Increment our counters	
				t += 1
				i += 1

			### Update our control values every m iterations
			sess.run([reset_Wh, reset_bh, reset_Wo, reset_bo, reset_batch_dbh, reset_batch_dWh, reset_batch_dbo, reset_batch_dWo], feed_dict = {x: trX, y: trY})
		
	return costs, variances


def plot_log():
	track_every = 10
	sgd_costs, sgd_variances = sgd_log(1000, track_every)
	svrg_costs, svrg_variances = svrg_log(1000, track_every)
	length = len(sgd_costs)
	plt.figure(1)
	plt.plot(track_every* np.arange(length), sgd_costs, color = 'blue', label = 'SGD')
	plt.plot(track_every* np.arange(length), svrg_costs, color = 'red', label = 'SVRG')
	plt.title("Cost")
	plt.legend(loc = 'upper right')
	plt.xlabel("Number of updates")

	plt.figure(2)
	plt.plot(track_every * np.arange(length), sgd_variances, color = 'blue', label = 'SGD')
	plt.plot(track_every * np.arange(length), svrg_variances, color = 'red', label = 'SVRG')
	plt.title("Variance of Update")
	plt.legend(loc = 'upper right')
	plt.xlabel("Number of updates")

	plt.show()

def plot_nn():
	track_every = 10

	i = 1
	for lr in [0.01, 0.05, 0.1, 0.2]:
		for batch_size in [1, 10, 100]:
			sgd_costs, sgd_variances = sgd_nn(1000, track_every, lr, batch_size)
			svrg_costs, svrg_variances = svrg_nn(1000, track_every, lr, batch_size)

			length = len(sgd_costs)
			plt.figure(i)
			plt.plot(track_every * np.arange(length), sgd_costs, color = 'blue', label = 'SGD')
			plt.plot(track_every *np.arange(length), svrg_costs, color = 'red', label = 'SVRG')
			plt.title("CE Loss- Batch size: {0}, Learning rate: {1}".format(batch_size, lr))
			plt.legend(loc = 'upper right')
			plt.xlabel("Number of updates")

			i += 1
			plt.figure(i)
			plt.plot(track_every * np.arange(length), sgd_variances, color = 'blue', label = 'SGD')
			plt.plot(track_every * np.arange(length), svrg_variances, color = 'red', label = 'SVRG')
			plt.title("Variance of Update- Batch size: {0}, Learning rate: {1}".format(batch_size, lr))
			plt.legend(loc = 'upper right')
			plt.xlabel("Number of updates")
			plt.yscale('log')

			i += 1

	plt.show()


plot_nn()