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

        word_ids = tf.keras.Input(shape=(None,), name='word_ids')
        words_embedding = tf.keras.layers.Embedding(
            input_dim=num_words, output_dim=args.we_dim, mask_zero=True, name='word_embeddings')(word_ids)

        charseqs = tf.keras.Input(shape=(None, None), name='charseqs')

        # Because cuDNN implementation of RNN does not allow empty sequences,
        # we need to consider only charseqs for valid words.
        # For CNN embeddings it is not strictly necessary, but we use it anyway.
        valid_words = tf.where(word_ids != 0, name='word_ids_not_0')
        cle = tf.gather_nd(charseqs, valid_words, name='charseqs_valid_words')

        chars_embedding = tf.keras.layers.Embedding(
            input_dim=num_chars, output_dim=args.cle_dim, mask_zero=True, name='chars_embedding')(cle)

        def cnn(width, cnn):
            cnn = tf.keras.layers.Conv1D(
                args.cnn_filters, width, strides=1, padding="valid", activation='relu', name='cnn' + str(width))(cnn)
            return tf.keras.layers.GlobalMaxPool1D(name='max_pool' + str(width))(cnn)

        cnn_concat = tf.keras.layers.Concatenate(name='concat_cnn_layers')(
            [cnn(width, chars_embedding) for width in range(2, args.cnn_max_width + 1)])

        def highway_network(x):
            T = tf.keras.layers.Dense(units=x.shape[1], activation='sigmoid', name='T_gate')(x)
            H = tf.keras.layers.Dense(units=x.shape[1], activation='relu', name='H')(x)
            C = tf.subtract(1.0, T, name='C_gate')
            return tf.add(tf.multiply(H, T), tf.multiply(x, C), name='highway_network')

        # Now we copy cle-s back to the original shape.
        cle = tf.scatter_nd(valid_words, highway_network(cnn_concat), [tf.shape(charseqs, name='shape0')[
                            0], tf.shape(charseqs, name='shape1')[1], highway_network(cnn_concat).shape[-1]], name='highway_net_scattered')

        # `tf.keras.layers.Concatenate()` layer preserves masks
        # (contrary to raw methods like tf.concat).
        we_cle_concat = tf.keras.layers.Concatenate(
            name='we_cle_concat')([words_embedding, cle])

        if args.rnn_cell == "LSTM":
            rnn_cell = tf.keras.layers.LSTM(
                args.rnn_cell_dim, return_sequences=True)
        elif args.rnn_cell == "GRU":
            rnn_cell = tf.keras.layers.GRU(
                args.rnn_cell_dim, return_sequences=True)
        brnn_global = tf.keras.layers.Bidirectional(
            layer=rnn_cell, merge_mode='sum')(we_cle_concat)

        predictions = tf.keras.layers.Dense(
            units=num_tags, activation=tf.keras.activations.softmax)(brnn_global)

        self.model = tf.keras.Model(inputs=[word_ids, charseqs], outputs=predictions)
        tf.keras.utils.plot_model(
            self.model, to_file='model.png', show_shapes=True, show_layer_names=True,
            rankdir='TB', expand_nested=False, dpi=96
        )
        self.model.compile(optimizer=tf.optimizers.Adam(),
                           loss=tf.losses.SparseCategoricalCrossentropy(),
                           metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")])

        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    def train_epoch(self, dataset, args):
        for batch in dataset.batches(args.batch_size):

            metrics = self.model.train_on_batch(
                x=[batch[dataset.FORMS].word_ids, batch[dataset.FORMS].charseqs],
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
        # We assume that model metric are already resetted at this point.
        for batch in dataset.batches(args.batch_size):
            metrics = self.model.test_on_batch(
                x=[batch[dataset.FORMS].word_ids, batch[dataset.FORMS].charseqs],
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
    parser.add_argument("--cnn_filters", default=16, type=int, help="CNN embedding filters per length.")
    parser.add_argument("--cnn_max_width", default=4, type=int, help="Maximum CNN filter width.")
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
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences, add_bow_eow=True)

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
