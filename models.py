import tensorflow as tf
import numpy as np

# class MNISTModel(object):
# 	def __init__(self, l2_reg):
# 		self.l2_reg = l2_reg

# 		self.input = tf.placeholder(tf.float32, [None, 784])
# 		self.targets = tf.placeholder(tf.float32, [None, 10])

# 		self.params = {}

# 	def compute_loss():


class LogisticRegression(object):
	def __init__(self, l2_reg):
		self.l2_reg = l2_reg

		self.input = tf.placeholder(tf.float32, [None, 784])
		self.targets = tf.placeholder(tf.float32, [None, 10])

		W = tf.Variable(tf.zeros([784, 10]), name = 'W')
		b = tf.Variable(tf.zeros([10]), name = 'b')
		self.params = {"W": W, "b": b}

		with tf.name_scope("control"):
			self.control_params = {}
			self.control_gradients = {}
			for param_name, param_var in self.params.items():
				self.control_params[param_name] = tf.Variable(tf.zeros_like(param_var.initialized_value()), name = param_name)
				self.control_gradients[param_var.name] = tf.Variable(tf.zeros_like(param_var.initialized_value()), name = "gradient_" + param_name)

		self.loss = self.compute_loss(self.params)
		self.control_loss = self.compute_loss(self.control_params)

		self.loss_summary = tf.summary.scalar("Loss", self.loss)

	def compute_loss(self, loss_params):
		pred_probs = tf.nn.softmax(tf.matmul(self.input, loss_params["W"]) + loss_params["b"])
		return tf.reduce_mean(-tf.reduce_sum(self.targets * tf.log(pred_probs), reduction_indices = 1)) + self.l2_reg * tf.nn.l2_loss(loss_params["W"])




class NeuralNetwork(object):
	def __init__(self, l2_reg):
		self.l2_reg = l2_reg

		self.input = tf.placeholder(tf.float32, [None, 784])
		self.targets = tf.placeholder(tf.float32, [None, 10])

		Wh = tf.Variable(tf.zeros([784, 100]), name = 'Wh')
		bh = tf.Variable(tf.zeros([100]), name = 'bh')
		Wo = tf.Variable(tf.zeros([100, 10]), name = 'Wo')
		bo = tf.Variable(tf.zeros([10]), name = 'bo')

		self.params = {"Wh": Wh, "bh": bh, "Wo": Wo, "bo": bo}

		with tf.name_scope("control"):
			self.control_params = {}
			self.control_gradients = {}
			for param_name, param_var in self.params.items():
				self.control_params[param_name] = tf.Variable(tf.zeros_like(param_var.initialized_value()), name = param_name)
				self.control_gradients[param_var.name] = tf.Variable(tf.zeros_like(param_var.initialized_value()), name = "gradient_" + param_name)

		self.loss = self.compute_loss(self.params)
		self.control_loss = self.compute_loss(self.control_params)

	def compute_loss(self, params):
		h = tf.nn.sigmoid(tf.matmul(self.input, params["Wh"]) + params["bh"])
		pred_probs = tf.nn.softmax(tf.matmul(h, params["Wo"]) + params["bo"])
		return tf.reduce_mean(-tf.reduce_sum(self.targets * tf.log(pred_probs), reduction_indices = 1)) + self.l2_reg * (tf.nn.l2_loss(params["Wh"]) + tf.nn.l2_loss(params["Wo"]))
