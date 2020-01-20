import sys, os
import argparse
import random
import numpy as np
import pandas as pd
import pickle
sys.path.append(
	os.path.join(
		os.path.dirname(__file__), 
		'..', 
		'utils'
	)
)
import Padronizar
from ErroPercentualAbsoluto import *
import Arquivo

from sklearn.metrics import r2_score
from sklearn.preprocessing   import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn                 import preprocessing
from cascade_correlation_network import CasCorNet
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from activation_functions import *


np.set_printoptions()

def add_bias(X):

	X_bias_incl = np.ones((X.shape[0], X.shape[1] + 1))
	X_bias_incl[:, :-1] = X
	
	return X_bias_incl

def load_and_preprocess_data():

	data = Arquivo.ler()
	x, y = Padronizar.dividir(data, 60, 12)
	minmaxscaler = MinMaxScaler(feature_range=(0,1))
	dataNX, listMin,  listMax  = Padronizar.normalizarLinear(x, 0.1, 0.9)
	dataNY, listMin1, listMax2 = Padronizar.normalizarLinear(y, 0.1, 0.9)
	X_train, X_test, y_train, y_test = train_test_split(dataNX,
														dataNY,
														train_size = 0.8,
														test_size  = 0.2)
	# X_train = 1.0 * X_train / 255
	# X_test = 1.0 * X_test / 255	

	return add_bias(X_train), y_train, add_bias(X_test), y_test

def main(args):

	X_train, y_train, X_test, y_test = load_and_preprocess_data()

	if args.time == 'train':
		
		input_size = len(X_train[0])
		output_size = len(np.unique(y_train))

		net = CasCorNet(input_size, output_size, args)
		net.set_data(X_train, y_train, X_test, y_test)
		net.train()

	elif args.time == 'test':
		
		try:
			net = pickle.load(open(args.resume, 'rb'))
		except:
			exit('Cascade correlation net could not be loaded from file, bye')

		net.train()

def exit(msg):
	print(msg)
	sys.exit(-1)


def sanity_checks(args):
	try:
		assert args.learning_rate >= 0
		assert args.learning_rate <= 1
	except: 
		exit('Learning rate should be between 0 and 1')

	try:
		assert args.patience < 0.1
	except:
		exit('Patience should be a small float')

	try:
		assert args.activation_func in func_dict.keys()
		args.activation_func = func_dict[args.activation_func]
	except:
		exit('Activation function should be in:' + str(func_dict.keys()))

	try:
		assert args.output_file is not None
	except:
		exit('Output file unknown.')

	try:
		assert args.time in ['train', 'test']
	except:
		exit('Time should be train|test')

	try:
		if args.time == 'test':
			assert args.resume is not None
	except:
		exit('Resume should be the name of the file in which the saved model is found if time=test')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.001)
	parser.add_argument('--minibatch-size', type=int, dest='minibatch_size', default=1)
	parser.add_argument('--patience', type=float, dest='patience', default=0.0001)
	parser.add_argument('--activation', type=str, dest='activation_func', default='sigmoid')
	parser.add_argument('--time', type=str, dest='time', default='train')
	parser.add_argument('--candidates', type=int, dest='num_candidates', default=5)
	parser.add_argument('--output-file', type=str, dest='output_file',default='output_file')
	parser.add_argument('--resume', type=str, dest='resume', default=None)

	args, unknown = parser.parse_known_args()
	sanity_checks(args)
	print(args)

	main(args)
