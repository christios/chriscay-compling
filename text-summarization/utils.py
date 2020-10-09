from geneeanlpclient import g3
import numpy as np
from nltk.corpus import stopwords
from nltk import edit_distance
from operator import itemgetter
import re
from string import punctuation
from typing import Union, Optional, Set, Dict, List, Any, Iterable, Tuple
from scipy.stats import norm
from math import exp

DOUBLE_QUOTES_RE = r"[«»“”„‟❝❞〝〞〟＂]"


def closest_match(word, wordlist):
    """
    Finds the closest match between a token's relations, and one of the tokens
    in the wordlist.
    """
    edit_distances = {candidate: edit_distance(
        word, candidate) for candidate in wordlist}
    return min(edit_distances.items(), key=itemgetter(1))[0]


def term_freq(wordlist: List[Tuple[str, List[str]]],
              cell_type: str,
              freq_list=None,
              rel=False) -> Dict[str, float]:
    """
    Computes the term-frequency of a word list and embeds information about
    relations into the the term frequency list. Tokens within the same sentence
    which are involved in a `Relation` will have 1 added to their counts to highlight
    their relationship.
    """
    tf: Dict[str, float] = dict()
    if cell_type == 'tfidf':
        for term in wordlist:
            tf[term[0]] = tf.get(term[0], 0.0) + 1.0
    # Term frequency is calculated w.r.t. the whole article and not individual sentences.
    elif cell_type == 'tf':
        for term in wordlist:
            tf[term[0]] = freq_list[term[0]]

    if rel:
        for term in wordlist:
            if term[1]:
                tf[term[0]] += 1
                wl = [w[0] for w in wordlist]
                for arg in term[1]:
                    if arg not in wl:
                        cm = closest_match(arg, wl)
                    else:
                        cm = arg
                    tf[cm] += 1
    return tf


def idf(n, docfreq):
    """ Computes the inverse document frequency """
    return np.log10(np.reciprocal(docfreq) * n)


def create_index(analysis: g3.Analysis) -> Tuple[Dict[str, g3.Entity], Dict[str, Dict[str, List[str]]]]:
    sent_to_token_to_rel: Dict[str, Dict[str, List[str]]] = {}
    for rel in analysis.relations:
        for support in rel.support:
            args = []
            sent_id = support.tokens[0].sentence._id
            for arg in rel.args:
                args.append(arg.name)
            sent_to_token_to_rel.setdefault(sent_id, {})[
                support.tectoToken.tokens[-1]._id] = args

    token_to_mention: Dict[str, g3.Entity] = {}
    for e in analysis.entities:
        for m in e.mentions:
            for t in m.tokens:
                token_to_mention[t._id] = m

    return token_to_mention, sent_to_token_to_rel


def read_doc(sentence_info,
             token_to_mention={},
             sent_to_token_to_rel={},
             entity=False,
             minimum=1,
             maximum=40,
             lang='english') -> Union[None, Tuple[str, List[Tuple[str, List[str]]]]]:
    """ If `entity` is set to False, splits a sentence into a word list of tuples (token and relation),
        else, it splits it into a word list of entites.
        Note: Disregards sentences with less than `minimum` and more than `maximum` words (30 words
        when not in entity mode)."""

    if minimum < len(sentence_info.tokens) < maximum:
        sw = stopwords.words(lang)
        sent_id: str = sentence_info._id
        words_temp: List[Tuple[str, List[str], str]] = []
        relations: Dict[str, List[str]] = sent_to_token_to_rel.get(sent_id)
        if entity:
            for t in sentence_info.tokens:
                if t._id in token_to_mention:
                    m = token_to_mention[t._id]
                    words_temp.append((m.mentionOf.stdForm,
                                       relations.get(t._id) if relations else None,
                                       m._id)
                                      )
                elif t.deepLemma not in sw and not any(p in t.deepLemma for p in punctuation):
                    words_temp.append((t.deepLemma,
                                       relations.get(t._id) if relations else None,
                                       None)
                    )
            # Keep only one token out of the ones which pertains to the same mention
            words_unique_mention: List[Tuple[str, List[str], str]] = []
            for word in words_temp:
                if word[2] and word[2] not in [w[2] for w in words_unique_mention]:
                    words_unique_mention.append(word)
                elif not word[2]:
                    words_unique_mention.append(word)
            words: List[Tuple[str, List[str]]] = [w[:-1] for w in words_unique_mention]
        else:
            words: List[Tuple[str, List[str]]] = [(t.deepLemma, relations.get(t._id) if relations else None)
                                                    for t in sentence_info.tokens if (t.deepLemma not in sw and
                                                    not any(p in t.deepLemma for p in punctuation))
            ]
    else:
        return
    
    return (sent_id, words)


def extract_sentences_cross(VT, S, sentences_raw, columnheader, n=10) -> Dict[str, str]:
    """
    - Returns a list of `n` sentences (represented by the columns of VT).
    - `k` concepts are used if k is provided else the number of concepts is
    the number of values in S which are >= p*max(S)
    """
    # (Ozsoy et al., 2010) set to zero all concept values (row by row) which are less than the
    # the average concept value (for that row) for each sentence.
    averages = np.average(VT, axis=1)
    for i in range(VT.shape[0]):
        for j in range(VT.shape[1]):
            if VT[i][j] < averages[i]:
                VT[i][j] = 0.
    # Set `k` to be the index at which the difference between two consecutive SVs
    # drops below the average of all differences between consecutive SVs.
    diff_S = [x-S[i+1] for i, x in enumerate(S[:-1])]
    avg_diff_S = sum(diff_S)/(S.size - 1)
    for i, diff in enumerate(diff_S):
        if diff < avg_diff_S:
            k = i
            break

    VTk, Sk = VT[:k, ], S[:k]
    sentences_score = np.add.reduce((np.square(VTk).T * Sk).T)
    # Generates a function f(x) `len_weights` which sharply (polynomial of order 6)
    # increases around x = 4 and which 'sigmoidally' decreases around 40, to give
    # gradually less weight to longer sentences.
    poly, sigmoid = list(range(100)), list(range(100))
    for i, x in enumerate(poly):
        poly[i] = -(0.06*(x-20))**6 + 1
        sigmoid[i] = 1/(1+exp((0.4*x-20)))
    len_weights = poly[:30] + sigmoid[32:] + [0.]*2

    # Exclude sentences which are indirect speech (which contain long quoted content)
    for sent_id, sentence in sentences_raw.items():
        sentence = re.sub(DOUBLE_QUOTES_RE, r'"', sentence)
        matches = re.findall(r'"(.*?)"', sentence)
        if matches:
            for match in matches:
                if len(match.split(' ')) > 8:
                    sentences_score[columnheader.index(sent_id)] = 0
                    break
        # For cases when segmentation is bad and we get half a quote in one sentence
        if sentence.count(r'"') == 1:
            sentences_score[columnheader.index(sent_id)] = 0
        # Weigh sentences according to their length
        sentences_score[columnheader.index(
            sent_id)] *= len_weights[len(sentence.split(' '))]

    sentences_score_sorted = sentences_score.argsort()[::-1][:n]
    sentences: Dict[str, str] = {columnheader[sent_index]: sentences_raw[columnheader[sent_index]]
                 for sent_index in sentences_score_sorted}

    return sentences
