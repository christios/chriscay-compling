#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

# Dataset for generating sequences, with labels predicting whether the cumulative sum
# is odd/even.
class Dataset:
    def __init__(self, sequences_num, sequence_length, sequence_dim, seed, shuffle_batches=True):
        sequences = np.zeros([sequences_num, sequence_length, sequence_dim], np.int32)
        labels = np.zeros([sequences_num, sequence_length, 1], np.bool)
        generator = np.random.RandomState(seed)
        for i in range(sequences_num):
            sequences[i, :, 0] = generator.randint(0, max(2, sequence_dim), size=[sequence_length])
            labels[i, :, 0] = np.bitwise_and(np.cumsum(sequences[i, :, 0]), 1)
            if sequence_dim > 1:
                sequences[i] = np.eye(sequence_dim)[sequences[i, :, 0]]
        self._data = {"sequences": sequences.astype(np.float32), "labels": labels}
        self._size = sequences_num

        self._shuffler = np.random.RandomState(seed) if shuffle_batches else None

    @property
    def data(self):
        return self._data

    @property
    def size(self):
        return self._size

    def batches(self, size=None):
        permutation = self._shuffler.permutation(self._size) if self._shuffler else np.arange(self._size)
        while len(permutation):
            batch_size = min(size or np.inf, len(permutation))
            batch_perm = permutation[:batch_size]
            permutation = permutation[batch_size:]

            batch = {}
            for key in self._data:
                batch[key] = self._data[key][batch_perm]
            yield batch


class Network:
    def __init__(self, args):
        # Construct the model.
        sequences = tf.keras.layers.Input(shape=[args.sequence_length, args.sequence_dim])

        if args.rnn_cell == "SimpleRNN":
            rnn_cell = tf.keras.layers.SimpleRNNCell(units=args.rnn_cell_dim)
        elif args.rnn_cell == "LSTM":
            rnn_cell = tf.keras.layers.LSTMCell(units=args.rnn_cell_dim)
        elif args.rnn_cell == "GRU":
            rnn_cell = tf.keras.layers.GRUCell(units=args.rnn_cell_dim)
        else:
            raise Exception("The input is not a valid RNN cell type")

        last_layer = tf.keras.layers.RNN(cell=rnn_cell, return_sequences=True, dtype=tf.float32)(sequences)

        if args.hidden_layer != 0:
            last_layer = tf.keras.layers.Dense(args.hidden_layer, activation=tf.nn.relu)(last_layer)

        predictions = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(last_layer)

        self.model = tf.keras.Model(inputs=sequences, outputs=predictions)

        self._optimizer = tf.keras.optimizers.Adam()
        self._loss = tf.losses.BinaryCrossentropy()
        self._metrics = {'loss': tf.metrics.Mean(), 'accuracy': tf.metrics.BinaryAccuracy()}
        # Create a summary file writer using `tf.summary.create_file_writer`.
        # I usually add `flush_millis=10 * 1000` arguments to get the results reasonably quickly.
        self._writer = tf.summary.create_file_writer(flush_millis=10*1000, logdir=args.logdir)

    @tf.function
    def train_batch(self, batch, clip_gradient):

        with tf.GradientTape() as gt:
            # Probabilities from self.model, passing `training=True` to the model
            outs = self.model(batch["sequences"], training=True)
            # Loss
            loss = self._loss(batch["labels"], outs)
            grads = gt.gradient(target=loss, sources=self.model.trainable_variables)
            if clip_gradient:
                grads, grad_norm = tf.clip_by_global_norm(grads, clip_gradient)
            else:
                grad_norm = tf.linalg.global_norm(grads)
            self._optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            tf.summary.experimental.set_step(self._optimizer.iterations)

            # Then, in the following with block, which records summaries
            # each 100 steps, perform the following:
            with self._writer.as_default(), tf.summary.record_if(self._optimizer.iterations % 100 == 0):
                # Iterate through the self._metrics
                for name, metric in self._metrics.items():
                    # Reset each metric:
                    # For "loss" metric, apply currently computed `loss`
                    if name == "loss":
                        tf.summary.scalar("train/" + name, loss)
                    # For other metrics, compute their value using the gold labels and predictions
                    else:
                        metric(batch["labels"], outs)
                        # Emit the summary to TensorBoard
                        tf.summary.scalar("train/" + name, metric.result())
                # Emit the gradient_norm
                tf.summary.scalar("train/gradient_norm", grad_norm)

    def train_epoch(self, dataset, args):
        for batch in dataset.batches(args.batch_size):
            self.train_batch(batch, args.clip_gradient)

    @tf.function
    def predict_batch(self, batch):
        return self.model(batch["sequences"], training=False)

    def evaluate(self, dataset, args):

        for name, metric in self._metrics.items():
            if name != "loss":
                metric.reset_states()
        results = []
        for batch in dataset.batches(args.batch_size):
            # For each, predict probabilities
            outs = self.predict_batch(batch)
            # Compute loss of the batch
            loss = self._loss(batch["labels"], outs)
            # Update the metrics (the "loss" metric uses current loss, others are computed
            # using the gold labels and the predictions)
            result = dict()
            result["loss"] = loss
            for name, metric in self._metrics.items():
                if name != "loss":
                    metric(batch["labels"], outs)
                    result[name] = metric.result()
            results.append(result)
        # Create a dictionary `metrics` with results, using names and values in `self._metrics`.
        metrics = dict()
        with self._writer.as_default():
            for name in self._metrics.keys():
                val = results[-1][name]
                # val = np.average(np.array([d[name] for d in results]))
                tf.summary.scalar("test/" + name, val, step=None)
                metrics[name] = val
        return metrics

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
    parser.add_argument("--clip_gradient", default=0., type=float, help="Norm for gradient clipping.")
    parser.add_argument("--hidden_layer", default=0, type=int, help="Additional hidden layer after RNN.")
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
    parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=10, type=int, help="RNN cell dimension.")
    parser.add_argument("--sequence_dim", default=1, type=int, help="Sequence element dimension.")
    parser.add_argument("--sequence_length", default=50, type=int, help="Sequence length.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--test_sequences", default=1000, type=int, help="Number of testing sequences.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--train_sequences", default=10000, type=int, help="Number of training sequences.")
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

    # Create the data
    train = Dataset(args.train_sequences, args.sequence_length, args.sequence_dim, seed=42, shuffle_batches=True)
    test = Dataset(args.test_sequences, args.sequence_length, args.sequence_dim, seed=43, shuffle_batches=False)

    # Create the network and train
    network = Network(args)
    for epoch in range(args.epochs):
        network.train_epoch(train, args)
        metrics = network.evaluate(test, args)
        print(metrics)
    with open("sequence_classification.out", "w") as out_file:
        print("{:.2f}".format(100 * metrics["accuracy"]), file=out_file)
