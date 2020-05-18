#!/usr/bin/env python3
#coding: utf-8

import sys
from collections import defaultdict
from collections import Counter
import math
import re
import pdb
import os
import urllib.request

# parameters
source_filename = 'MADAR.corpus26.MSA-preproc_TOK_TAG.conllu'
source_raw_filename = 'MADAR.corpus26.MSA-preproc'
target_filename = 'MADAR.corpus26.Beirut'
alignment_filename = 'MSA-Beirut.f'

URL = "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/arabic-padt-ud-2.5-191206.udpipe?sequence=8&isAllowed=y"
path = os.path.basename(URL)
if not os.path.exists(path):
	print("Downloading data {}...".format(path))
	urllib.request.urlretrieve(URL, filename=path)

# number of sentences -- in PUD it is always 1000
SENTENCES = 1000

# field indexes
ID = 0
FORM = 1
LEMMA = 2
UPOS = 3
XPOS = 4
FEATS = 5
HEAD = 6
DEPREL = 7

l1, l2 = 0.9, 0.9
parts_of_speech = [	'ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'SYM', 'VERB', 'ADP', 'AUX', 'CCONJ', 
					'DET', 'X', 'NUM', 'PART', 'PRON', 'SCONJ', 'PUNCT']

def read_alignment(fh):
	""" Returns dict[source_id] = [target_id_1, target_id_2, target_id_3...]
 		and a reverse one as well
	"""
	line = fh.readline()
	src2tgt, tgt2src = defaultdict(list),  defaultdict(list)
	for st in line.split():
		(src, tgt) = st.split('-')
		src, tgt = int(src), int(tgt)
		src2tgt[src].append(tgt); tgt2src[tgt].append(src)
	return (src2tgt, tgt2src)

def read_sentence(fh_source, fh_target, target=False, source=None):
	""" - Returns a list of tokens, where each token is a list of fields;
		- ID is switched from 1-based to 0-based to match alignment style
		- Some preprocessing is done to ensure the integrity of the data:
			- 	The parser (while working on the source language) often splits
				a sentence from the parallel corpus to two or more sentences. This function
				joins these split sentences into the same one.
			- 	Sometimes two sentences on different lines in the parallel corpus are joined,
				so this function just excludes these sentences from our analysis as they aren't many
				and were very diffucult to deal with during preprocessing.
	"""
	# If the function is called with the target file handler
	if target:
		sentence = fh_target.readline()
		# Exclude punctutation to make token dictionary assembly more consistent
		sentence_no_punct = sentence.translate(str.maketrans('', '', '؟،‎.')).split()
		# Keep a copy of the punctuated sentence for future use
		sentence = re.sub(r'([؟،‎\.])', r' \1', sentence).split()

	else:
		current_sentence_raw, current_sentence = source_raw.readline(), ""
		split_current, join_current, offset = False, False, 0
		sentence = list()
		for line in fh_source:
			# End of sentence (if '\n' or if we are not in join mode)
			if line == '\n' and not join_current:
				# Exclude punctutation to make token dictionary assembly more consistent
				sentence_no_punct = [fields for fields in sentence if fields[FORM] not in "؟،‎."]
				# Renumber IDs in increasing order as they have been messed up by deleting components of compound tokens
				for i in range(len(sentence_no_punct)):
					sentence_no_punct[i][ID] = i
				break
			# Beginning of sentence
			elif line.startswith('#'):
				
				if line.find('text') != -1:
					# Source sentence (which might be longer or shorter than the one 
					# in the parallel corpus depending on what the UDpipe model did)
					current_sentence += line[9:]
					if current_sentence != current_sentence_raw:
						# If it is shorter than its counterpart in the parallel corpus, then stop accepting
						# '# text = ...' lines (until we join the sentence) by entering join mode (using join_current)
						if len(current_sentence) < len(current_sentence_raw):
							eol = current_sentence.find('\n')
							current_sentence = current_sentence[:eol] + ' ' + current_sentence[eol+1:]
							join_current = True
						# If it is longer than its counterpart in the parallel corpus, then simply skip the sentence
						# by entering split mode (using split_current) which disregards all token lines in the CoNLLU file, and raw sentences
						# in the parallel corpus until the raw sentence is reconstituted
						else:
							current_sentence_raw = current_sentence_raw[:-1]
							while(current_sentence[:-1] != current_sentence_raw):
								fh_target.readline()
								current_sentence_raw += (' ' + source_raw.readline()[:-1])
							split_current = True
					else:
						join_current = False
				else:
					continue
			# Middle of sentence
			else:
				# If in split mode, then disregard all tokens until source sentence is reconstituted
				if not split_current:
					fields = line.strip().split('\t')
					# If the token is simple
					if fields[ID].isdigit():
						# make IDs 0-based to match alignment IDs
						fields[ID] = int(fields[ID]) - 1
						sentence.append(fields)
					# If the token is compound
					elif re.search(r'\d+-\d+', fields[ID]):
						# Throw away the split tokens, keep the compound one, and assign the POS
						# of the second split token to the compound one because the second in Arabic is
						# usually a content word while the first one is a stop word
						# NOTE: reindexing is done before returning
						sentence.append(fields)
						fh_source.readline()
						fields[UPOS] = fh_source.readline().strip().split('\t')[UPOS]
				else:
					continue
	return sentence_no_punct, sentence

def print_sent (sentence):
	for line in source_sentence:
		print(line)
	print('')

def print_dict (dictionary):
	for k, v in smoothed_prob_1_1.items():
		print("{}: {}".format(k, v))

raw_proj_1_1, raw_proj_1_n = {}, {}
smoothed_prob_1_1, smoothed_prob_1_n = {}, {}

with open(source_filename) as source, open(target_filename) as target, open(alignment_filename) as alignment, open(source_raw_filename) as source_raw:
	for sentence_id in range(SENTENCES):
		(src2tgt, tgt2src) = read_alignment(alignment)
		source_sentence, source_sentence_punc = read_sentence(fh_source=source, fh_target=target, source=source_raw)
		target_sentence, target_sentence_punc = read_sentence(fh_source=source, fh_target=target, target=True)
		
		#If two lines from the source corpus were joined by the parser, drop them
		if not source_sentence:
			read_alignment(alignment)
			continue

		# Starting here, I follow the guidelines of the Yarowsky and Ngai (2001) paper
		# I create a dictionary of the form: dict[word_type] = {'ADJ': 0, ..., 'PUNCT': 0}
		# with the values of the inner dictionaries being the amount of times the word_type
		# was assigned a particular POS from the UD tagset.
		# The paper suggests to treat cases of one-to-one and one-to-n alignments separately
		# in order to perform weighted smoothing later because usually one-to-n alignments are
		# a considerable source of inaccuracy.
		# The projections are simply transferred as per the alignment, thus the "raw" term.
		for source_id, target_ids in src2tgt.items():
			# For cases where the alignment is one-to-one (store in raw_proj_1_1)
			if len(target_ids) == 1:
				target_form = target_sentence[target_ids[0]]
				source_POS = source_sentence[source_id][UPOS]
				POS_vec = raw_proj_1_1.setdefault(target_form, {'ADJ': 0, 'ADV': 0, 'INTJ': 0, 'NOUN': 0, 'PROPN': 0, 'SYM': 0,
					'VERB': 0, 'ADP': 0, 'AUX': 0, 'CCONJ': 0, 'DET': 0, 'X': 0,
					'NUM': 0, 'PART': 0, 'PRON': 0, 'SCONJ': 0, 'PUNCT': 0})
				POS_vec[source_POS] += 1
			# For cases where the alignment is one-to-n (store in raw_proj_1_n)
			elif len(target_ids) > 1:
				for target_id in target_ids:
					target_form = target_sentence[target_id]
					source_POS = source_sentence[source_id][UPOS]
					POS_vec = raw_proj_1_n.setdefault(target_form, {'ADJ': 0, 'ADV': 0, 'INTJ': 0, 'NOUN': 0, 'PROPN': 0, 'SYM': 0,
						'VERB': 0, 'ADP': 0, 'AUX': 0, 'CCONJ': 0, 'DET': 0, 'X': 0,
						'NUM': 0, 'PART': 0, 'PRON': 0, 'SCONJ': 0, 'PUNCT': 0})
					POS_vec[source_POS] += 1

# Aggressively moothed probabilities are assembled here using the l1 parameter by keeping the two most frequent tags
# and setting the probability of the rest to zero.
for target_form, POS_vec in raw_proj_1_1.items():
	most_freq_POS1 = max(POS_vec, key=lambda key: POS_vec[key])
	POS_vec_without_max = {POS: POS_vec[POS] for POS in POS_vec if POS not in {most_freq_POS1}}
	most_freq_POS2 = max(POS_vec_without_max, key=lambda key: POS_vec_without_max[key])
	prob2 = l1 * POS_vec[most_freq_POS2] / sum(POS_vec.values())
	prob1 = 1 - prob2
	smoothed_prob_1_1[target_form] = {most_freq_POS1: prob1, most_freq_POS2: prob2}
# The same is done for the 1_n dictionary
for target_form, POS_vec in raw_proj_1_n.items():
	most_freq_POS1 = max(POS_vec, key=lambda key: POS_vec[key])
	POS_vec_without_max = {POS: POS_vec[POS] for POS in POS_vec if POS not in {most_freq_POS1}}
	most_freq_POS2 = max(POS_vec_without_max, key=lambda key: POS_vec_without_max[key])
	prob2 = l1 * POS_vec[most_freq_POS2] / sum(POS_vec.values())
	prob1 = 1 - prob2
	smoothed_prob_1_n[target_form] = {most_freq_POS1: prob1, most_freq_POS2: prob2}


def Pt_w (POS, form):
	sp11 = smoothed_prob_1_1[form].get(POS, 0) if smoothed_prob_1_1.get(form) else 0
	sp1n = smoothed_prob_1_n[form].get(POS, 0) if smoothed_prob_1_n.get(form) else 0
	return l2 * sp11 + (1 - l2) * sp1n

def write_sentence(sentence):
	"""	- Takes list of lists as input, i.e., as returned by read_sentence()
		- Switches ID back to 1-based and converts them to strings
		- Joins fields by tabs and tokens by endlines and returns the CONLL string
    """
	result = list()
	for i, token in enumerate(sentence):
        # Switch back to 1-based IDs
		if token in "؟،‎.":
			POS_max = 'PUNCT'
		else:
			maximum, POS_max = 0, ''
			for POS in parts_of_speech:
				val = Pt_w(POS, token)
				if val > maximum:
					maximum, POS_max = val, POS

		fields = "{}\t{}\t{}\t{}\t_\t_\t_\t_\t_\t_".format(i + 1, token, token, POS_max)
		result.append(fields)
	result.append('')
	return '\n'.join(result)

# Repeat same process as before to write the sentences with the tags
with open(source_filename) as source, open(target_filename) as target, open(alignment_filename) as alignment, open(source_raw_filename) as source_raw:
	for sentence_id in range(SENTENCES):
		(src2tgt, tgt2src) = read_alignment(alignment)
		source_sentence, source_sentence_punc = read_sentence(fh_source=source, fh_target=target, source=source_raw)
		target_sentence, target_sentence_punc = read_sentence(fh_source=source, fh_target=target, target=True)
		
		#If two lines from the source corpus were joined by the parser, drop them
		if not source_sentence:
			read_alignment(alignment)
			continue

		print(write_sentence(target_sentence_punc))

#######################################################################################################################################################

# Token list for the target corpus; remove punctuation first
tokens_target = []
with open(target_filename) as fh_target:
	for sentence in fh_target:
		sentence_no_punct = sentence.translate(str.maketrans('', '', "؟،‎.")).split()
		for token in sentence_no_punct:
			tokens_target.append(token)

# Frequency list for the target corpus
tokens_target_fl = dict(Counter(tokens_target).most_common())
T = len(tokens_target)
Pw = {token : freq/T for token, freq in tokens_target_fl.items()}

# The rest of these functions were written to train a bigram tagger as described by the paper
# but due to time constraints I could not complete them
def Pw_t (form, POS):
	return Pt_w(POS, form) * Pw[form] / sum(Pt_w(POS, form_x) * Pw[form_x] for form_x in tokens_target_fl)

def PW_T (W, T):
	if len(W) == len(T):
		return math.prod(Pw_t(w, t) for w, t in zip(W, T))
	else:
		raise "Word sequence (W) length does not match tag sequence (T) length"

