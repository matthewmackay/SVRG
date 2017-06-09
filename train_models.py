import tensorflow as tf
import numpy as np
from models import LogisticRegression, NeuralNetwork
from datetime import datetime

# Set-up
# ==================================================
now = datetime.utcnow().strftime("%m.%d.%H.%M.%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_float("l2_reg", 1e-4, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 10, "Batch Size (default: 10)")
tf.flags.DEFINE_integer("num_updates", 1000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("update_control", 10, "Number of updates to run before updating control parameters")
tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model's loss on training set after this many steps (default: 10)")
tf.flags.DEFINE_float("initial_lr", 0.05, "Initial learning rate of the model")
#tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
#tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data loading
# ==================================================
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
train_inputs, train_targets, test_inputs, test_targets = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
num_train = train_inputs.shape[0]

def batch_iter():
	pass


# Set-up
# ==================================================
svrg_flag = True
now = datetime.utcnow().strftime("%m.%d.%H.%M.%S")
root_logdir = "tf_logs"
if (svrg_flag):
	svrg_or_sgd = "SVRG"
else:
	svrg_or_sgd = "SGD"
logdir = "{}/{}-{}/".format(root_logdir,svrg_or_sgd, now)

# Debugging utilities
# ==================================================
# computation flows from parents to children
def parents(op):
  return set(input.op for input in op.inputs)

def children(op):
  return set(op for out in op.outputs for op in out.consumers())

def get_graph():
  """Creates dictionary {node: {child1, child2, ..},..} for current
  TensorFlow graph. Result is compatible with networkx/toposort"""

  ops = tf.get_default_graph().get_operations()
  return {op: children(op) for op in ops}


def print_tf_graph(graph):
  """Prints tensorflow graph in dictionary form."""
  for node in graph:
    for child in graph[node]:
      print("%s -> %s" % (node.name, child.name))


# Training
# ==================================================
with tf.Graph().as_default():
	sess = tf.Session()
	with sess.as_default():
		model = LogisticRegression(l2_reg = FLAGS.l2_reg)

        ### Define Training procedure
		global_step = tf.Variable(0, name = "global_step", trainable=False)
		optimizer = tf.train.GradientDescentOptimizer(FLAGS.initial_lr)
		grads_vars = optimizer.compute_gradients(model.loss, model.params.values())
		

		#control_grads_vars = optimizer.compute_gradients(model.control_loss, model.control_params.values())
		[dcontrolW, dcontrolb] = tf.gradients(model.control_loss, [model.control_params['W'], model.control_params['b']])
		control_grads_vars = [(dcontrolW, model.control_params['W']), (dcontrolb, model.control_params['b'])]

		dW, db = tf.gradients(model.loss, [model.params["W"], model.params["b"]])

        ### Formulate gradients for SVRG update
		svrg_grads_vars = []
		control_vars_to_grad = dict([(var.name, grad) for (grad, var) in control_grads_vars])
		for (grad, var) in grads_vars:
				svrg_grad = grad
				svrg_grad -= control_vars_to_grad["control/" + var.name]
				svrg_grad += model.control_gradients[var.name]
				svrg_grads_vars.append((svrg_grad, var))

        ### Can apply either a regular SGD update or a SVRG update
		sgd_update = optimizer.apply_gradients(grads_vars, global_step = global_step)
		svrg_update = optimizer.apply_gradients(svrg_grads_vars, global_step = global_step)

        ### Operations required to reset values of control variables
		# reset_control_vars = []
		# reset_control_grads = []

		# for param_name, param_var in model.params.items():
		# 	reset_control_vars.append(model.control_params[param_name].assign(param_var))
		# 	reset_control_grads.append(model.control_gradients[param_var.name].assign(control_vars_to_grad["control/" + param_var.name]))

		reset_W = model.control_params["W"].assign(model.params["W"])
		reset_b = model.control_params["b"].assign(model.params["b"])
		reset_control_vars = [reset_W, reset_b]

		w_name = model.params['W'].name
		b_name = model.params['b'].name
		reset_dW = model.control_gradients[w_name].assign(dW)
		reset_db = model.control_gradients[b_name].assign(db)
		reset_control_grads = [reset_dW, reset_db]

		file_writer = tf.summary.FileWriter(logdir, sess.graph) # for 1.0

        ### Initialize all variables
		sess.run(tf.global_variables_initializer())

		print(svrg_update)
		for t in reset_control_grads:
			print(parents(t.op))

        ### Always run a SGD update to start
		curr_batch = {model.input: train_inputs[:FLAGS.batch_size, :], model.targets: train_targets[:FLAGS.batch_size, :]}
		_, update_num = sess.run([sgd_update, global_step], curr_batch)

		while(update_num < FLAGS.num_updates):
			start_batch = (update_num * FLAGS.batch_size) % num_train
			end_batch = start_batch + FLAGS.batch_size
			curr_batch = {model.input: train_inputs[start_batch:end_batch,:], model.targets: train_targets[start_batch:end_batch,:]}
			
			if (svrg_flag):
				if (update_num - 1 % FLAGS.update_control == 0):
					sess.run(reset_control_vars + reset_control_grads, feed_dict = {model.input: train_inputs, model.targets: train_targets})
				
					spec_control_W = sess.run(model.control_params['W'])
					spec_W = sess.run(model.params['W'])
					assert np.allclose(spec_control_W, spec_W)

					spec_dW = sess.run(dW, feed_dict = {model.input: train_inputs, model.targets: train_targets})
					spec_control_dW = sess.run(model.control_gradients[model.params['W'].name])
					assert np.allclose(spec_dW, spec_control_dW)


				_, update_num = sess.run([svrg_update, global_step], feed_dict = curr_batch)
			
			else:
				_, update_num = sess.run([sgd_update, global_step], feed_dict = curr_batch)
			
			if (update_num % FLAGS.evaluate_every == 0):
				curr_loss, summary_str = sess.run([model.loss, model.loss_summary], feed_dict = {model.input: train_inputs, model.targets: train_targets})
				print("Update Number:{0}\t, Loss:{1}".format(update_num, curr_loss))
				file_writer.add_summary(summary_str, update_num)

		file_writer.close()