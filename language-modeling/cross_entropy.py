#!/usr/bin/env python3

from collections import Counter
from itertools import islice
from math import log2

def linint (filename, encoding = 'utf-8'):

	# Imports the text file
	fh = open(filename, encoding = encoding)
	text = fh.read()
	fh.close()

	# Unigram tokens list
	tokens = text.splitlines()
	tokens_len = len(tokens)

	# Segmentation of the data
	test = tokens[-20000:]				# Test data
	heldout = tokens[-60000:-20000]		# Heldout data
	train = tokens[:-60000]				# Training data

	# Inserting blank tokens in the beginning of the training, test, and heldout data
	train.insert(0, "<s>"); train.insert(0, "<s>");
	heldout.insert(0, "<s>"); heldout.insert(0, "<s>");
	test.insert(0, "<s>"); test.insert(0, "<s>");

	# Frequency lists of unigrams/bigrams/trigrams for training/heldout/test data
	train_bi_fl = dict(Counter(zip(train, islice(train, 1, None))).most_common())
	train_tri_fl = dict(Counter(zip(train, islice(train, 1, None), islice(train, 2, None))).most_common())
	train_uni_fl = dict(Counter(train).most_common())
	test_tri_fl = dict(Counter(zip(test, islice(test, 1, None), islice(test, 2, None))).most_common())
	
	# Lists of unigrams/bigrams/trigrams for training/heldout/test data
	train_tri = list(zip(train, islice(train, 1, None), islice(train, 2, None)))
	heldout_tri = list(zip(heldout, islice(heldout, 1, None), islice(heldout, 2, None)))
	test_tri = list(zip(test, islice(test, 1, None), islice(test, 2, None)))

	T = len(train) 					# Training data text size
	V = len(train_uni_fl)			# Training data vocabulary size

	# Computation of the training data probability distributions for the uniform, unigram, bigram, and trigram models
	P0_uni = 1/V
	P1_uni = {unigram : freq/T for unigram, freq in train_uni_fl.items()}
	P2_bi = {bigram : freq/train_uni_fl[bigram[0]] for bigram, freq in train_bi_fl.items()}
	P3_tri = {trigram : freq/train_bi_fl[trigram[:2]] for trigram, freq in train_tri_fl.items()}

	# Setting initial values for EM algorithm
	l1 = 0.25; l2 = 0.25; l3 = 0.25; l0 = 0.25;
	l0_prev = 0; l1_prev = 0; l2_prev = 0; l3_prev = 0;

	# Function for the computation of the smoothed trigram model
	def Ps (trigram):
		return l3*P3_tri.get(trigram, 0) + l2*P2_bi.get(trigram[1:], 0) + l1*P1_uni.get(trigram[2], 0) + l0*P0_uni
	
	# Implementation of the EM algorithm (USING TRAINING DATA)
	while abs(l0 - l0_prev) > 0.0001 or abs(l1 - l1_prev) > 0.0001 or abs(l2 - l2_prev) > 0.0001 or abs(l3 - l3_prev) > 0.0001: 

		# If a unigram/bigram/trigram in the heldout data does not exist in the training data, then the probability of it
		# is set to 0 in this procedure
		c_l0 = sum((l0 * P0_uni)/Ps(trigram) 					for trigram in train_tri)
		c_l1 = sum((l1 * P1_uni.get(trigram[2], 0))/Ps(trigram) for trigram in train_tri)
		c_l2 = sum((l2 * P2_bi.get(trigram[1:], 0))/Ps(trigram) for trigram in train_tri)
		c_l3 = sum((l3 * P3_tri.get(trigram, 0))/Ps(trigram) 	for trigram in train_tri)

		l0_prev = l0; l1_prev = l1; l2_prev = l2; l3_prev = l3;

		l0 = c_l0/(c_l0 + c_l1 + c_l2 + c_l3)
		l1 = c_l1/(c_l0 + c_l1 + c_l2 + c_l3)
		l2 = c_l2/(c_l0 + c_l1 + c_l2 + c_l3)
		l3 = c_l3/(c_l0 + c_l1 + c_l2 + c_l3)

	print("Implementation of the EM algorithm (using training data):")
	print('%.5f'%l0, '%.5f'%l1, '%.5f'%l2, '%.5f'%l3, sep = '\t', end='')

	# Resetting initial values for EM algorithm
	l1 = 0.25; l2 = 0.25; l3 = 0.25; l0 = 0.25;
	l0_prev = 0; l1_prev = 0; l2_prev = 0; l3_prev = 0;

	# Implementation of the EM algorithm (USING HELDOUT DATA)
	while abs(l0 - l0_prev) > 0.0001 or abs(l1 - l1_prev) > 0.0001 or abs(l2 - l2_prev) > 0.0001 or abs(l3 - l3_prev) > 0.0001: 

		# If a unigram/bigram/trigram in the heldout data does not exist in the training data, then the probability of it
		# is set to 0 in this procedure
		c_l0 = sum((l0 * P0_uni)/Ps(trigram) 					for trigram in heldout_tri)
		c_l1 = sum((l1 * P1_uni.get(trigram[2], 0))/Ps(trigram) for trigram in heldout_tri)
		c_l2 = sum((l2 * P2_bi.get(trigram[1:], 0))/Ps(trigram) for trigram in heldout_tri)
		c_l3 = sum((l3 * P3_tri.get(trigram, 0))/Ps(trigram) 	for trigram in heldout_tri)

		l0_prev = l0; l1_prev = l1; l2_prev = l2; l3_prev = l3;

		l0 = c_l0/(c_l0 + c_l1 + c_l2 + c_l3)
		l1 = c_l1/(c_l0 + c_l1 + c_l2 + c_l3)
		l2 = c_l2/(c_l0 + c_l1 + c_l2 + c_l3)
		l3 = c_l3/(c_l0 + c_l1 + c_l2 + c_l3)

	# Cross entropy calculation
	crossH = (-1/len(test)) * sum (log2(Ps(trigram)) for trigram in test_tri)
	print("Implementation of the EM algorithm (using heldout data)")
	print('%.5f'%l0, '%.5f'%l1, '%.5f'%l2, '%.5f'%l3, '%.5f'%crossH, sep = '\t', end='')

	# Setting the test values
	P0 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
	P1 = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
	l0_test = 0; l1_test = 0; l2_test = 0; l3_test = 0;
	
	def Ps_test (trigram):
		return l3_test*P3_tri.get(trigram, 0) + l2_test*P2_bi.get(trigram[1:], 0) + l1_test*P1_uni.get(trigram[2], 0) + l0_test*P0_uni

	for p in P0:
		l3_test = l3 + p*(1-l3)
		l2_test = l2 - l2*(p*(1-l3))/(1-l3)
		l1_test = l1 - l1*(p*(1-l3))/(1-l3)
		l0_test = l0 - l0*(p*(1-l3))/(1-l3)
		# Cross entropy calculation
		crossH = (-1/len(test)) * sum (log2(Ps_test(trigram)) for trigram in test_tri)
		print('%.5f'%l0_test, '%.5f'%l1_test, '%.5f'%l2_test, '%.5f'%l3_test, '%.5f'%crossH, sep = '\t')

	print("")

	for p in P1:
		l3_test = l3 - l3*(1-p)
		l2_test = l2 + l2*l3*(1-p)/(1-l3)
		l1_test = l1 + l1*l3*(1-p)/(1-l3)
		l0_test = l0 + l0*l3*(1-p)/(1-l3)
		# Cross entropy calculation
		crossH = (-1/len(test)) * sum (log2(Ps_test(trigram)) for trigram in test_tri)
		print('%.5f'%l0_test, '%.5f'%l1_test, '%.5f'%l2_test, '%.5f'%l3_test, '%.5f'%crossH, sep = '\t')

	# Training set coverage calculation
	cov = sum(1 for token in test if token in train_uni_fl)/len(test)
	print("\n", "Coverage (", filename, "): ", '%.3f'%cov)


linint('TEXTEN1.txt')					# Tests with all cases ran for the English text
print("\n")
linint('TEXTCZ1.txt', 'iso-8859-2')		# Tests with all cases ran for the Czech text
