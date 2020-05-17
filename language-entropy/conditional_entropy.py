#!/usr/bin/env python3

from collections import Counter
from itertools import islice
from random import random, uniform
from math import log2

# Messup likelihood list
MUL = [	0,
		0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,
		0.0001,	0.0001,	0.0001,	0.0001,	0.0001,	0.0001,	0.0001,	0.0001,	0.0001,	0.0001,
		0.001,	0.001,	0.001,	0.001,	0.001,	0.001,	0.001,	0.001,	0.001,	0.001,
		0.01,	0.01,	0.01,	0.01,	0.01,	0.01,	0.01,	0.01,	0.01,	0.01,
		0.05,	0.05,	0.05,	0.05,	0.05,	0.05,	0.05,	0.05,	0.05,	0.05,
		0.1, 	0.1,	0.1,	0.1,	0.1,	0.1,	0.1,	0.1,	0.1,	0.1]

mode = "entropy"

def mul (filename, encoding = 'utf-8'):
	# Imports the text file
	fh = open(filename, encoding = encoding)
	text = fh.read()
	fh.close()

	# Unigram tokens list
	tokens = text.splitlines()
	tokens_len = len(tokens)
	
	# Distinct tokens list
	tokens_list = list(dict(Counter(tokens).most_common()).keys())

	# Character frequency list of characters without the end of line character
	char_list = list(dict(Counter([char for char in text]).most_common()).keys())
	char_list = char_list[1:]
	char_list_len = len(char_list)

	# Character messup
	print("Character messup:")

	for mul in MUL:

		# Create a list of messed up tokens with messup likelihood (mul)
		tokens_mul = tokens.copy()
		for token in range(len(tokens_mul)):
			token_mul_list = list(tokens_mul[token])
			for char in range(len(token_mul_list)):
				if random() <= mul:
					token_mul_list[char] = char_list[int(uniform(0, char_list_len))]
			tokens_mul[token] = "".join(token_mul_list)

		# Token frequency list with <s> inserted at the beginning
		tokens_mul.insert(0, "<s>")
		tokens_mul_fl = dict(Counter(tokens_mul).most_common())

		# Bigram frequency list
		# Creates a frequency list of all bigrams in the text from most to least frequent
		# by aggregating a list of words to the same list shited by 1 using zip
		bigram_mul_fl = dict(Counter(zip(tokens_mul, islice(tokens_mul, 1, None))).most_common())

		# Bigram probability list P(i,j)
		N = sum(bigram_mul_fl.values(), 0.0)	# Total number of bigrams (or tokens) as a float
		Pij = {bigram : freq/N for bigram, freq in bigram_mul_fl.items()}

		# Bigram conditional probability list P(j|i) using MLE
		cPji = {bigram : freq/tokens_mul_fl[bigram[0]] for bigram, freq in bigram_mul_fl.items()}

		# Conditional entropy calculation
		cHji = -1 * sum(Pij[k] * log2(cPji[k]) for k in cPji)

		if mode == "entropy":
			print(cHji)
		elif mode == "perplexity":
			print(2**cHji)

	# Word messup
	print("\nWord messup:")
	for mul in MUL:

		# Create a list of messed up tokens with messup likelihood (mul)
		tokens_mul = tokens.copy()
		for token in range(len(tokens_mul)):
			if random() <= mul:
				tokens_mul[token] = tokens_list[int(uniform(0, len(tokens_list)))]	#Previous mistake: used the original text. Correct way: use list of distinct words

		# Token frequency list
		tokens_mul.insert(0, "<s>");
		tokens_mul_fl = dict(Counter(tokens_mul).most_common())

		# Bigram frequency list
		# Creates a frequency list of all bigrams in the text from most to least frequent
		# by aggregating a list of words to the same list shited by 1 using zip
		bigram_mul_fl = dict(Counter(zip(tokens_mul, islice(tokens_mul, 1, None))).most_common())

		# Bigram probability list P(i,j)
		N = sum(bigram_mul_fl.values(), 0.0)	# Total number of bigrams (or tokens) as a float
		Pij = {bigram : freq/N for bigram, freq in bigram_mul_fl.items()}

		# Bigram conditional probability list P(j|i) using MLE
		cPji = {bigram : freq/tokens_mul_fl[bigram[0]] for bigram, freq in bigram_mul_fl.items()}

		# Conditional entropy calculation
		cHji = -1 * sum(Pij[k] * log2(cPji[k]) for k in cPji)

		if mode == "entropy":
			print(cHji)
		elif mode == "perplexity":
			print(2**cHji)

	info = {}
	info['wc'] = tokens_len						# Word count
	info['cc'] = len(text)						# Character count
	info['ccpw'] = len(text)/tokens_len			# Character count per word
	freq_list = Counter(tokens).most_common()	# Frequency list of tokens
	info['vc'] = len(freq_list)					# Vocabulary size
	info['mf1'] = freq_list[0] 					# Most frequent word frequency
	info['mf2'] = freq_list[1]					# Second most frequent word frequency
	info['mf3'] = freq_list[2]					# ...
	info['mf4'] = freq_list[3]
	info['mf5'] = freq_list[4]
	info['mf6'] = freq_list[5]
	info['wcf1'] = sum(unigram[1] for unigram in freq_list if unigram[1] == 1)	# Frequency of words with frequency 1

	return info

print(mul('TEXTEN1.txt'))					# Tests with all cases ran for the English text
print(mul('TEXTCZ1.txt', 'iso-8859-2'))		# Tests with all cases ran for the Czech text

