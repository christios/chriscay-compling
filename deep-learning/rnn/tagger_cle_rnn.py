#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from morpho_dataset import MorphoDataset

class Network:
    def __init__(self, args, num_words, num_tags, num_chars):

        word_ids = tf.keras.Input(shape=(None,))
        words_embedding = tf.keras.layers.Embedding(input_dim=num_words, output_dim=args.we_dim, mask_zero=True)(word_ids)

        # The RNN character-level embeddings utilize the input `charseqs`
        # containing a sequence of character indices for every input word.
        # Again, padded characters have index 0.
        charseqs = tf.keras.Input(shape=(None, None))
        
        # Because cuDNN implementation of RNN does not allow empty sequences,
        # we need to consider only charseqs for valid words.
        valid_words = tf.where(word_ids != 0)
        cle = tf.gather_nd(charseqs, valid_words)

        chars_embedding = tf.keras.layers.Embedding(input_dim=num_chars, output_dim=args.cle_dim, mask_zero=True)(cle)
        gru_cell = tf.keras.layers.GRU(units=args.cle_dim, return_sequences=False)
        brnn_cle = tf.keras.layers.Bidirectional(layer=gru_cell, merge_mode='concat')(chars_embedding)
        
        # Now we copy cle-s back to the original shape.
        cle = tf.scatter_nd(valid_words, brnn_cle, [tf.shape(charseqs)[0], tf.shape(charseqs)[1], brnn_cle.shape[-1]])

        we_cle_concat = tf.keras.layers.Concatenate()([words_embedding, cle])

        if args.rnn_cell == "LSTM":
            rnn_cell = tf.keras.layers.LSTM(args.rnn_cell_dim, return_sequences=True)
        elif args.rnn_cell == "GRU":
            rnn_cell = tf.keras.layers.GRU(args.rnn_cell_dim, return_sequences=True)
        brnn_global = tf.keras.layers.Bidirectional(layer=rnn_cell, merge_mode='sum')(we_cle_concat)

        predictions = tf.keras.layers.Dense(units=num_tags, activation=tf.keras.activations.softmax)(brnn_global)

        self.model = tf.keras.Model(inputs=[word_ids, charseqs], outputs=predictions)
        self.model.compile(optimizer=tf.optimizers.Adam(),
                           loss=tf.losses.SparseCategoricalCrossentropy(),
                           metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")])
        
        tf.keras.utils.plot_model(
            self.model, to_file='tagger_cle_rnn.png', show_shapes=True, show_layer_names=True,
            rankdir='TB', expand_nested=False, dpi=96
        )
        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    def train_epoch(self, dataset, args):
        for batch in dataset.batches(args.batch_size):
            metrics = self.model.train_on_batch(x=[batch[dataset.FORMS].word_ids, batch[dataset.FORMS].charseqs],
                                                y=batch[dataset.TAGS].word_ids,
                                                reset_metrics=True
                                                )
            # Generate the summaries each 100 steps
            if self.model.optimizer.iterations % 100 == 0:
                tf.summary.experimental.set_step(self.model.optimizer.iterations)
                with self._writer.as_default():
                    for name, value in zip(self.model.metrics_names, metrics):
                        tf.summary.scalar("train/{}".format(name), value)

    def evaluate(self, dataset, dataset_name, args):
        for batch in dataset.batches(args.batch_size):
            metrics = self.model.test_on_batch(x=[batch[dataset.FORMS].word_ids, batch[dataset.FORMS].charseqs],
                                               y=batch[dataset.TAGS].word_ids,
                                               reset_metrics=False
                                               )
        self.model.reset_metrics()

        metrics = dict(zip(self.model.metrics_names, metrics))
        with self._writer.as_default():
            tf.summary.experimental.set_step(self.model.optimizer.iterations)
            for name, value in metrics.items():
                tf.summary.scalar("{}/{}".format(dataset_name, name), value)

        return metrics


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
    parser.add_argument("--cle_dim", default=32, type=int, help="CLE embedding dimension.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--max_sentences", default=5000, type=int, help="Maximum number of sentences to load.")
    parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=64, type=int, help="RNN cell dimension.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences)

    # Create the network and train
    network = Network(args,
                      num_words=len(morpho.train.data[morpho.train.FORMS].words),
                      num_tags=len(morpho.train.data[morpho.train.TAGS].words),
                      num_chars=len(morpho.train.data[morpho.train.FORMS].alphabet))
    for epoch in range(args.epochs):
        network.train_epoch(morpho.train, args)
        metrics = network.evaluate(morpho.dev, "dev", args)

    metrics = network.evaluate(morpho.test, "test", args)
    with open("tagger_we.out", "w") as out_file:
        print("{:.2f}".format(100 * metrics["accuracy"]), file=out_file)
