# Copyright 2020 Christian Cayralat

# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http: // www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import numpy as np
import re
import json
from collections import Counter
from typing import Union, Optional, Set, Dict, List, Any, Iterable, Tuple

import utils

from geneeanlpclient import g3
from geneea.sdkplus.client import AdvancedClient
from geneea.sdkplus.request import RequestBuilder

Document = Tuple[str, Tuple[str, List[str]]]
DocumentsRaw = Dict[str, str]
LANGUAGE = {'english': 'en',
            'dutch': 'nl'
            }


def prepare_data(infile: str = '',
                 test: str = '',
                 save: bool = False,
                 entity: bool = False,
                 minimum: int = 1,
                 maximum: int = 40,
                 lang: str = 'english'
                 ) -> Tuple[List[Document], DocumentsRaw, g3.Analysis]:
    """
    - Extracts sentences from the input file
    - documents only holds sentences which will be used in the A matrix
    - sentences_raw[sent_id] = raw_sentence (origText)
    """

    assert infile or test, f"Either need to have an input file or be in test mode"
    assert not (
        infile and test), f"Can't input a file while in test mode. Remove the '-i' flag."

    documents: List[Document] = []
    sentences_raw: DocumentsRaw = {}
    if infile:
        client = AdvancedClient(
            apiHost='http://alpha.g', apiPath=AdvancedClient.apiPathForWorkflow('prototype-full-analysis-news'),
            batchSize=5, threadCount=4,
        )
        with open(infile) as fh:
            rq_builder = RequestBuilder.fullAnalysis(
                language=LANGUAGE[lang], domain='news')
            analysis = client.analyzeOne(rq_builder.build(text=fh.read()))
        if save:
            with open(save, 'w') as fout:
                print(json.dumps(g3.toDict(analysis),
                                 ensure_ascii=False), file=fout)
    elif test:
        with open(test, 'r') as fp:
            analysis = g3.reader.fromDict(json.load(fp))

    token_to_mention, sent_to_token_to_rel = utils.create_index(analysis)
    data_split = [sent for sent in analysis.paragraphs[0].sentences]
    for sentence in data_split:
        # Here a document is a sentence, and `sent` is a tuple as so: sent = (snet_id, wordlist)
        sent = utils.read_doc(sentence, token_to_mention, sent_to_token_to_rel,
                              entity=entity, minimum=minimum, maximum=maximum, lang=lang)
        if sent:
            documents.append(sent)
            sentences_raw[sentence._id] = sentence.origText

    return documents, sentences_raw, analysis


def prepare_A_matrix(documents, relations, cell_type='tfidf', rel=False):
    # Term Frequency
    freq_list = Counter([word[0] for doc in documents for word in doc[1]])
    tf: List[Tuple[str, Dict[str, float]]] = [(sentence[0], utils.term_freq(
        sentence[1], cell_type, freq_list, rel=rel)) for sentence in documents]

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
    dfvector = np.sum(np.array([docfreqmatrix(sentence[1])
                                for sentence in tf]), axis=0)

    #Create term-frequency matrix with corresponding headers
    tf = [(sentence[0], termfreqmatrix(sentence[1])) for sentence in tf]
    A = [sentence[1] for sentence in tf]
    columnheader = [sentence[0] for sentence in tf]
    rowheader = vocabulary

    # Preparing the matrices (tfidf from tf matrix and idf vector)
    A = np.array(A).T   # Rows are words and columns are the sentences
    if cell_type and cell_type == 'tfidf':
        idfvector = utils.idf(len(columnheader), dfvector)
        idfvector = np.array(np.transpose(idfvector))
        idfvector = np.reshape(idfvector, (-1, 1))
        A = A * idfvector

    return A, rowheader, columnheader


if __name__ == "__main__":
    count = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', dest='infile',  type=str,
                        help='Path of the input file to the script.')
    parser.add_argument('-eval', '--evaluation', dest='evaluation',  type=str,
                        help='Path of the reference file to compare against. Computes the Rouge1 score')
    parser.add_argument('-t', '--test', dest='test',  type=str,
                        help='Path of an article\'s analysis file to use instead of the article itself. Cannot be used at the same time with -i.')
    parser.add_argument('-s', '--save', dest='save',  type=str,
                        help='If an analysis on an input file was performed, saves it. To use in conjunction with -i.')
    parser.add_argument('-n', dest='n',  type=int, default=10,
                        help='Number of sentences used in the summary')
    parser.add_argument('-c', '--cell-type', dest='cell_type',  type=str, default='tfidf',
                        help='Type of values to be used for the cells of the A matrix. Current options: tf, tfidf')
    parser.add_argument('-lang', '--language', dest='lang',  type=str, default='english',
                        help='Language of the article (no format, just the name of the language).')
    parser.add_argument('-l', '--length', dest='length',  type=str, default='10,40',
                        help='Length of sentences to consider for the A matrix; format: min_length,max_length.')
    parser.add_argument('-e', '--entity', default=False, action="store_true",
                        help="Entities will be included in the computation of the A matrix.")
    parser.add_argument('-rel', '--relations', default=False, action="store_true",
                        help="Relations will be included in the computation of the A matrix. Relations are note supported in Dutch")
    parser.add_argument('-p', '--print', default=False, action="store_true",
                        help="Print the chosen sentences to the default output stream.")
    args = parser.parse_args()
    min_length, max_length = tuple([int(x) for x in args.length.split(',')])

    documents, sentences_raw, analysis = prepare_data(infile=args.infile,
                                                      test=args.test,
                                                      save=args.save,
                                                      entity=args.entity,
                                                      minimum=min_length,
                                                      maximum=max_length,
                                                      lang=args.lang)
    A, rowheader, columnheader = prepare_A_matrix(documents=documents,
                                                  relations=analysis.relations,
                                                  cell_type=args.cell_type,
                                                  rel=args.relations)
    # Singular Value Decomposition on the tfidf matrix
    U, S, VT = np.linalg.svd(A, full_matrices=0)
    sentences = utils.extract_sentences_cross(VT=VT,
                                              S=S,
                                              sentences_raw=sentences_raw,
                                              columnheader=columnheader,
                                              n=args.n)
    # Sort the sentences by the order in which they appear in the text
    sentences = sorted(sentences.items(), key=lambda sent: sent[0])
    
    if args.print:
        for i, (sent_id, sent) in enumerate(sentences):
            # print('[Sentence '+str(i+1)+'] :\t'+str(sent))
            print(str(sent))
        print('\n')

    if args.evaluation:
        from sumeval.metrics.rouge import RougeCalculator
        with open(args.evaluation) as fh:
            reference = fh.read()
        rouge = RougeCalculator(stopwords=True, lang='en')
        rouge_1 = rouge.rouge_n(
            summary='\n'.join([sent[1] for sent in sentences]),
            references=reference,
            n=1
        )
        rouge_2 = rouge.rouge_n(
            summary='\n'.join([sent[1] for sent in sentences]),
            references=reference,
            n=2
        )
        rouge_l = rouge.rouge_l(
            summary='\n'.join([sent[1] for sent in sentences]),
            references=reference
        )
        print(f'ROUGE1: {rouge_1}', f'ROUGE2: {rouge_2}', f'ROUGEL: {rouge_l}', sep='\n')
