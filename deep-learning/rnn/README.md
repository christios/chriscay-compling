# Recurrent Neural Network Applications and Implementations

## Neural Lemmatizer
* `lemmatizer_att.py`: Lemmatizer network implementation suing stacked RNNs with an attention mechanism
* `lemmatizer_task.py`: main of `lemmatizer_att.py`

## Neural Tagger
* `tagger_cle_rnn.py`: Tagger which uses both word and character-level embeddings, and bidirectional RNNs with memory units.

## Sequence Classification with Gradient Clipping
* `sequence_classification.py`: The network processes sequences of 50 small integers and computes parity for each prefix of the sequence. Summaries are written to Tensorboard to observe the effect of the <i>exploding gradient</i> problem.

## Speech Recognition
* `speech_recognition.py`: Mel-frequency cepstral coefficients are fed into the network for each input clip. Then the CTC loss is computed alongside beam-search decoding.
