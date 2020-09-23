import sys
import argparse
import numpy as np
from utils import (term_freq, idf, read_doc, extract_sentences, create_index)
import re
import random

import datetime
import json
from geneeanlpclient import g3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', dest='infile',  type=str,
                        help='The input file to the script.')
    parser.add_argument('--api-key', dest='api_key',  type=str,
                        help='API key for the g3 client.')
    parser.add_argument('-t', '--test', dest='test',  type=str,
                        help='If used, then analysis is loaded and not performed.')
    parser.add_argument("--sentences", "-s", default=True, action="store_true",
                        help="Extract key sentences.")
    parser.add_argument("--entity", "-e", default=False, action="store_true",
                        help="Rows of the A matrix will represent extracted entities instead of tokens.")
    args = parser.parse_args()

    USER_KEY = args.api_key

    assert not (args.infile and args.test), f"Can't input a file while in test mode. Remove the '-i' flag."
    if args.infile:
        assert args.api_key, f"Need an API key for the g3 client to analyze the data"
    # Extracts sentences from the input file
    documents, data_split_raw = [], {}
    if args.infile:
        with open(args.infile) as fh, g3.Client.create(userKey=USER_KEY) as analyzer:
            requestBuilders = {
                'high-recall': g3.Request.Builder(
                    langDetectPrior='en',
                    analyses=[g3.AnalysisType.ALL],
                    referenceDate=datetime.datetime.now(),
                    returnMentions=True,
                    returnItemSentiment=True,
                    domain='high-recall')
            }
            data = fh.read()
            for domain, rb in requestBuilders.items():
                analysis = analyzer.analyze(rb.build(text=data))
    elif args.test:
        with open('analysis.json', 'r') as fp:
            analysis = g3.reader.fromDict(json.load(fp))
    
    token_to_entity = create_index(analysis)
    data_split = [sent for sent in analysis.paragraphs[0].sentences]
    for sentence in data_split:
        sent = read_doc(sentence, token_to_entity, entity=args.entity) # Here a document is a sentence
        if sent:
            documents.append(sent)
            data_split_raw[sentence._id] = sentence.origText

    # Term Frequency
    tf = [(sentence[0], term_freq(sentence[1])) for sentence in documents]

    # Get the vocabulary of the documents
    vocabulary = []
    for sentence in tf:
        for word in sentence[1]:
            if word not in vocabulary:
                vocabulary.append(word)
    vocabulary = np.array(vocabulary)

    def termfreqmatrix(tfdict):
        return [tfdict.get(word, 0) for word in vocabulary]

    def docfreqmatrix(tfdict):
        return [1.0 if (tfdict.get(word, 0) > 0) else 0. for word in vocabulary]

    #Create document-requency vector
    dfvector = np.sum(np.array([docfreqmatrix(sentence[1]) for sentence in tf]), axis=0)

    #Create term-frequency matrix
    tf = sorted([(sentence[0], termfreqmatrix(sentence[1])) for sentence in tf], key=lambda x: x[0])
    A = [sentence[1] for sentence in tf]
    columnheader = [sentence[0] for sentence in tf]
    rowheader = vocabulary

    # Preparing the matrices(tfidf from tf matrix and idf vector)
    A = np.array(A).T   # Rows are words and columns are the sentences
    idfvector = idf(len(columnheader), dfvector)
    idfvector = np.array(np.transpose(idfvector))
    idfvector = np.reshape(idfvector, (-1,1))
    A = A * idfvector

    # Singular Value Decomposition on the tfidf matrix
    # Summary Sentences - Extraction
    U, S, VT = np.linalg.svd(A, full_matrices=0)
    sentences = extract_sentences(VT, S, data_split_raw, columnheader)
    # for i,concept in enumerate(concepts):
    for i,sent in enumerate(sentences):
        print ('[Sentence '+str(i+1)+'] :\t'+str(sent)) #Final Summary
    print ('\n')
