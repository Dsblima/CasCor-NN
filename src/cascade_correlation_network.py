import numpy as np
import pickle
import sys, os
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from activation_functions import *
from hidden_units_pool import hiddenUnitsPool

colors = ['red', 'green', 'pink', 'blue', 'orange', 'magenta', 'olive']

class CasCorNet(object):
	def __init__(self, input_size, output_size, args):

		# I = input size, O = output size
		self.I = input_size
		self.O = output_size

		# initialize the weights
		# self.weights = self.init_weights()

		self.hidden_units = []

		# hyperparameters
		self.f = args.activation_func
		self.alpha = args.learning_rate
		self.mb_sz = args.minibatch_size
		self.eps = args.patience
		self.output_file = args.output_file

		# TODO
		self.max_iterations_io = 50
		self.max_iterations = 30
		self.eval_every = 10

		self.train_loss = np.array([])
		self.test_loss = np.array([])
		self.loss_figure = plt.figure()
		self.accuracy_figure = plt.figure()
		self.cm_figure = plt.figure()
		self.limit_points_X_train_local = []
		self.limit_points_ys = []

	def set_data(self, X_train, y_train, X_test, y_test):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test

	def init_weights(self):
		val = 1
		weights = np.random.uniform(-val, val, (self.O, self.I))
		print(self.O)
		print(self.I)

		print("weights")
		exit("pesos")
		#print(weights)
		return weights

	def accuracy(self, X_train_local, ts):
		_, ys = self.forward(X_train_local)
		acc = 1.0 * np.sum(np.argmax(ts, axis=1) == np.argmax(ys, axis=1)) / len(X_train_local)
		return acc

	def forward(self, X_train_local):
		hs = np.dot(X_train_local, self.weights[:,:len(X_train_local[0])].T)
		ys = self.f(hs)
		print(X_train_local.shape)
		exit("teste3")
		print(ys.shape)

		return hs, ys

	def backward(self, X_train_local, ts, ys):

		# initialize the gradient with zeros
		dweights = np.zeros(self.weights.shape)
		
		# add the gradients for all examples
		for i in range(len(ts)):
			delta = -(ts[i] - ys[i]) * self.f(ys[i], True)
			dweights += np.outer(delta, X_train_local[i])

		# take the average of the gradients for all examples
		dweights /= self.mb_sz

		return dweights

	def get_loss(self, mini_y_train_local, mini_ys, return_sum):
		if return_sum == True:
			print("teste 21")
			print(mini_y_train_local.shape)
			print(mini_ys.shape)
			return np.sum(0.5 * (mini_y_train_local - mini_ys) ** 2) / len(mini_y_train_local)
		else:
			return 0.5 * (mini_y_train_local - mini_ys)**2 / len(mini_y_train_local)

	def update_weights(self, dweights, loss):
		self.weights -= self.alpha * dweights

	def plot_loss(self, loss_val, loss_name):

		if loss_name == 'train':
			loss_arr = self.train_loss
		else:
			loss_arr = self.test_loss

		new_loss = np.zeros((loss_arr.shape[0] + 1))
		new_loss[:-1] = loss_arr
		new_loss[-1] = loss_val

		if loss_name == 'train':
			self.train_loss = new_loss
		else:
			self.test_loss = new_loss
		
		plt.figure(self.loss_figure.number)	
		plt.xlabel('Iteration')
		plt.ylabel('Loss')		
		h1, = plt.plot(self.train_loss, color='red', label='Training loss')
		h2, = plt.plot(np.arange(0, self.eval_every * len(self.test_loss), self.eval_every), self.test_loss, color='blue', label='Test loss')
		plt.scatter(self.limit_points_X_train_local, self.limit_points_ys, color='black', label='New hidden unit recruited')
		plt.legend(handles=[h1, h2])
		plt.savefig('training_loss.png')
		plt.clf()

	def plot_accuracy(self, train_accuracy, test_accuracy):

		if not hasattr(self, 'train_accuracies'):
			self.train_accuracies = [train_accuracy]
			self.test_accuracies = [test_accuracy]
		else:
			self.train_accuracies.append(train_accuracy)
			self.test_accuracies.append(test_accuracy)

		plt.figure(self.accuracy_figure.number)
		h1, = plt.plot(self.train_accuracies, color='red', label='Train accuracy')
		h2, = plt.plot(self.test_accuracies, color='blue', label='Test accuracy')
		plt.legend(handles=[h1, h2])
		plt.savefig('accuracy.png')
		plt.clf()

	def check_io_convergence(self, iteration):
		if iteration == self.max_iterations_io:
			self.converged = True
		elif len(self.train_loss) >= 2 and \
			 abs(self.train_loss[-1] - self.train_loss[-2]) < self.eps:
			self.converged = True

	def eval_network(self, X_test_local, ts_test, y_test_local):

		print('learning rate', self.alpha)

		for hidden_unit in self.hidden_units:
			vs = hidden_unit.get_best_candidate_values(X_test_local)
			print('X_train_local shape before', X_test_local.shape)
			X_test_local = self.augment_input(X_test_local, vs)
			print('X_train_local shape after', X_test_local.shape)

		self.hidden_units = []

		hs, ys = self.forward(X_test_local)
		test_loss = self.get_loss(ts_test, ys, True)

		# make predictions
		predictions = np.argmax(ys, axis=1)

		accuracy = 0.0
		confusion_matrix = np.zeros((self.O, self.O))
		for i in range(len(X_test_local)):
			confusion_matrix[y_test_local[i], predictions[i]] += 1.0
			if y_test_local[i] == predictions[i]:
				accuracy += 1

		accuracy /= len(X_test_local)
		print('Accuracy on test data: %.3f' % (accuracy))

		plt.figure(self.cm_figure.number)
		plt.imshow(confusion_matrix)
		plt.savefig('confusion_matrix.png')
		plt.clf()

		self.plot_loss(test_loss, 'test')

		return X_test_local

	def train_io(self, X_train_local, X_test_local, y_test_local, y_train_local):

		iteration = 0
		
		while not self.converged:

			shuffled_range =list(range(len(X_train_local)))
			# print(shuffled_range)
			np.random.shuffle(shuffled_range)

			total_loss = 0.0

			# get minibatches of data			
			for i in range(len(X_train_local) // self.mb_sz):

				indices = shuffled_range[i * self.mb_sz:(i+1) * self.mb_sz]
				# print(y_train_local.shape)
				mini_X_train_local = X_train_local[indices]
				# print(y_train_local.head())
				mini_y_train_local = y_train_local[indices]
				# sys.exit(-1)

				# forward pass
				mini_hs, mini_ys = self.forward(mini_X_train_local)
				
				# compute total loss on this minibatch
				loss = self.get_loss(mini_y_train_local, mini_ys, True)

				total_loss += loss

				# backward pass
				dweights = self.backward(mini_X_train_local, mini_y_train_local, mini_ys)

				# update the weights using delta rule
				self.update_weights(dweights, loss)
				
			train_accuracy = self.accuracy(X_train_local, y_train_local)
			test_accuracy = self.accuracy(X_test_local, y_test_local)

			print('TRAIN ACC=', train_accuracy)
			print('TEST ACC=', test_accuracy)
			
			self.plot_loss(total_loss / (len(X_train_local) / self.mb_sz), 'train')
			self.plot_accuracy(train_accuracy, test_accuracy)
			self.check_io_convergence(iteration)


			if (len(self.train_loss) - 1) % self.eval_every == 0:
				X_test_local = self.eval_network(X_test_local, ts_test, y_test_local)

			iteration += 1

		self.limit_points_X_train_local.append(len(self.train_loss))
		self.limit_points_ys.append(self.train_loss[-1])

		print('Input-output convergence after %d iterations' % iteration)

		return X_test_local

	def train(self):

		X_train_local = self.X_train
		y_train_local = self.y_train
		X_test_local = self.X_test
		y_test_local = self.y_test

		N, M = len(X_train_local), len(X_test_local) 
		# print(self.X_train.shape)
		# sys.exit(-1)

		# create target arrays with 1 for the correct class
		# ts = np.zeros((N, self.O))
		# ts[np.arange(N), y_train_local] = 1
		# ts_test = np.zeros((M, self.O))
		# ts_test[np.arange(M), y_test_local] = 1

		iteration = 0
		acceptable_loss = 0.01
		max_iterations = 100

		train_acc = []
		test_acc = []

		while True:
			# train the input-output connections until convergence
			self.converged = False

			X_test_local = self.train_io(X_train_local, X_test_local, y_test_local, y_train_local)
			self.X_test = X_test_local

			# measure loss on data and stop if it's acceptable
			_, ys = self.forward(X_train_local)
			losses = self.get_loss(ts, ys, False)

			loss = np.sum(losses)
			if loss < acceptable_loss:
				break

			if iteration == max_iterations:
				break

			# otherwise add a new hidden unit
			X_train_local = self.add_hidden_unit(X_train_local, ts, losses)
			self.X_train = X_train_local

			iteration += 1

			if iteration % 5 == 0:
				with open(self.output_file, 'wb') as f:
					pickle.dump(self, f)

	def augment_input(self, X_train_local, vs):
		new_X_train_local = np.zeros((X_train_local.shape[0], X_train_local.shape[1] + 1))
		new_X_train_local[:, :-1] = X_train_local
		new_X_train_local[:, -1] = vs

		return new_X_train_local

	def add_hidden_unit(self, X_train_local, ts, losses):
		
		# initialize a pool of 10 candidates # TODO: make param instead of 10
		candidates_pool = hiddenUnitsPool(self.I, self.O, 5)
		candidates_pool.train(X_train_local, losses)		
		vs = candidates_pool.get_best_candidate_values(X_train_local)
		X_train_local = self.augment_input(X_train_local, vs)

		self.hidden_units.append(candidates_pool)
		self.I += 1 	# just added one more element for each input, so the size fo the input has increased

		new_weights = self.init_weights()
		new_weights[:, :-1] = self.weights
		self.weights = new_weights

		return X_train_local