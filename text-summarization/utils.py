import numpy as np
from nltk.corpus import stopwords
import re


def term_freq(wordlist):
    """
    Computes the term-frequency of a word list
    """
    tf = dict()
    for term in wordlist:
        tf[term] = tf.get(term, 0.0) + 1.0
    return tf

def idf(n, docfreq):
    """ Computes the inverse document frequency """
    return np.log10(np.reciprocal(docfreq) * n)

def create_index(analysis):
    token_to_entity = {}
    for e in analysis.entities:
        for m in e.mentions:
            for t in m.tokens:
                token_to_entity.setdefault(t._id, []).append(e)
    return token_to_entity

def read_doc(sentence_info, token_to_entity={}, entity=False):
    """ If entity is set to False, splits a sentence into a word list of tokens,
        else, it splits it into a word list of entites.
        Note: Disregards sentences with less than 10 and more than 40 words (30 words
        when not in entity mode)."""
    sent_id = sentence_info._id
    if entity:
        if 10 < len(sentence_info.tokens) < 40:
            entities = []
            for t in sentence_info.tokens:
                if t._id in token_to_entity:
                    entities.append(token_to_entity[t._id] if isinstance(
                        token_to_entity[t._id], str) else token_to_entity[t._id][0])
            words = [e.stdForm for e in entities]
            return (sent_id, words)
    else:
        sw = stopwords.words('english')
        if 10 < len(sentence_info.tokens) < 30:
            words = [t.lemma for t in sentence_info.tokens if t.lemma not in sw]
            return (sent_id, words)

def extract_sentences(VT, S, data_split_raw, columnheader, k=5, n=10):
    """
    Returns a list of k concepts (represented by the rows of VT). Each concept is
    a list of n sentences which are the most prominent sentences related to that concept.
    """
    concepts = []
    sentences_len = np.add.reduce((np.square(VT[:k, ]).T * S[:k]).T)
    sentences_len_sorted = sentences_len.argsort()[::-1]
    sentences = (data_split_raw[columnheader[sentence_id]]
                 for sentence_id in sentences_len_sorted)
    # for _ in range(k):
    keysentences = []
    for _ in range(n):
        keysentences.append(next(sentences))
    # concepts.append(keysentences)
    return keysentences
