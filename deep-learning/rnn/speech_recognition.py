#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from timit_mfcc import TimitMFCC

class Network:
    def __init__(self, args):
        self._beam_width = args.ctc_beam

        inputs = tf.keras.Input(shape=[None, TimitMFCC.MFCC_DIM])

        rnn_cell = tf.keras.layers.LSTM(
            units=args.rnn_cell_dim,
            return_sequences=True,
        )
        brnn = tf.keras.layers.Bidirectional(
            layer=rnn_cell, merge_mode='sum')(inputs)
        brnn = tf.keras.layers.Dropout(rate=0.5)(brnn)
        input_residual = brnn
        for _ in range(1, args.brnn_layers):
            brnn = tf.keras.layers.Bidirectional(
                layer=rnn_cell, merge_mode='sum')(brnn)
            brnn = tf.keras.layers.Dropout(rate=0.5)(brnn)
            brnn = tf.add(brnn, input_residual)
        outputs = tf.keras.layers.Dense(
            len(TimitMFCC.LETTERS) + 1, activation=None)(brnn)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        tf.keras.utils.plot_model(
            self.model, to_file='speech_recognition.png', show_shapes=True, show_layer_names=True,
            rankdir='TB', expand_nested=False, dpi=96
        )

        lr_schedule = args.lr
        if args.decay:
            lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=args.lr,
                decay_steps=timit.train.size/args.batch_size*args.epochs
            )
        self._optimizer = tf.optimizers.Adam(lr_schedule)
        self._loss = self._ctc_loss  # tf.losses.SparseCategoricalCrossentropy()
        self._metrics = {"loss": tf.metrics.Mean(
        ), "edit_distance": tf.metrics.Mean()}
        self._writer = tf.summary.create_file_writer(
            args.logdir, flush_millis=10 * 1000)

    # Converts given tensor with `0` values for padding elements
    # to a SparseTensor.
    def _to_sparse(self, tensor):
        tensor_indices = tf.where(tf.not_equal(tensor, 0))
        return tf.sparse.SparseTensor(tensor_indices, tf.gather_nd(tensor, tensor_indices), tf.shape(tensor, tf.int64))

    # Convert given sparse tensor to a (dense_output, sequence_lengths).
    def _to_dense(self, tensor):
        tensor = tf.sparse.to_dense(tensor, default_value=-1)
        tensor_lens = tf.reduce_sum(
            tf.cast(tf.not_equal(tensor, -1), tf.int32), axis=1)
        return tensor, tensor_lens

    # Compute logits given input mfcc, mfcc_lens and training flags.
    # Also transpose the logits to `[time_steps, batch, dimension]` shape
    # which is required by the following CTC operations.
    def _compute_logits(self, mfcc, mfcc_lens, training):
        """ - training is only relevant when using dropout (if True -> training; False -> inference)
            - mask ("call" argument): Binary tensor of shape[batch, timesteps] indicating whether a
                given timestep should be masked(optional, defaults to None).
            - mask = list of mfcc examples (equivalent to one batch) varying in length (each unit
                is a time step) due to varying length of audio clips
        """
        logits = self.model(mfcc, mask=tf.sequence_mask(
            mfcc_lens), training=training)
        return tf.transpose(logits, [1, 0, 2])

    # Compute CTC loss using given logits, their lengths, and sparse targets.
    def _ctc_loss(self, logits, logits_len, sparse_targets):
        loss = tf.nn.ctc_loss(sparse_targets, logits, None,
                              logits_len, blank_index=len(TimitMFCC.LETTERS))
        self._metrics["loss"](loss)
        return tf.reduce_mean(loss)

    # Perform CTC predictions given logits and their lengths.
    def _ctc_predict(self, logits, logits_len):
        (predictions,), _ = tf.nn.ctc_beam_search_decoder(
            logits, logits_len, beam_width=self._beam_width)
        return tf.cast(predictions, tf.int32)

    # Compute edit distance given sparse predictions and sparse targets.
    def _edit_distance(self, sparse_predictions, sparse_targets):
        edit_distance = tf.edit_distance(
            sparse_predictions, sparse_targets, normalize=True)
        self._metrics["edit_distance"](edit_distance)
        return edit_distance

    @tf.function(experimental_relax_shapes=True)
    def train_batch(self, mfcc, mfcc_lens, targets):
        """ gt.gradient() arguments:
                - target: a list or nested structure of Tensors to be differentiated.
                - sources: a list or nested structure of Tensors. target will be differentiated against elements in sources.
        """
        with tf.GradientTape() as gt:
            logits = self._compute_logits(mfcc, mfcc_lens, training=True)
            loss = self._ctc_loss(logits, mfcc_lens, self._to_sparse(targets))
            grads = gt.gradient(
                target=loss, sources=self.model.trainable_variables)
            if args.clip_gradient:
                grads, grad_norm = tf.clip_by_global_norm(
                    grads, args.clip_gradient)
            else:
                grad_norm = tf.linalg.global_norm(grads)
            self._optimizer.apply_gradients(
                zip(grads, self.model.trainable_variables))
            tf.summary.experimental.set_step(self._optimizer.iterations)

            with self._writer.as_default(), tf.summary.record_if(self._optimizer.iterations % 100 == 0):
                tf.summary.scalar("train/gradient_norm", grad_norm)

        predicts = self._ctc_predict(logits, mfcc_lens)
        ED = tf.reduce_mean(self._edit_distance(
            predicts, self._to_sparse(targets)))
        return {"loss": loss, "edit_distance": ED}

    def train_epoch(self, dataset, args):
        results = []
        for batch in dataset.batches(args.batch_size):
            results.append(self.train_batch(
                batch["mfcc"], batch["mfcc_len"], batch["letters"]))

        CTC_loss = np.average(np.array([result['loss'] for result in results]))
        ED = np.average(np.array([result['edit_distance']
                                  for result in results]))
        return CTC_loss, ED

    @tf.function(experimental_relax_shapes=True)
    def evaluate_batch(self, mfcc, mfcc_lens, targets):
        logits = self._compute_logits(mfcc, mfcc_lens, training=False)
        predicts = self._ctc_predict(logits, mfcc_lens)
        ED = tf.reduce_mean(self._edit_distance(
            predicts, self._to_sparse(targets)))
        CTC_loss = self._ctc_loss(logits, mfcc_lens, self._to_sparse(targets))
        return{"loss": CTC_loss, "edit_distance": ED}

    def evaluate(self, dataset, dataset_name, args):
        results = []
        for batch in dataset.batches(args.batch_size):
            results.append(self.evaluate_batch(
                batch["mfcc"], batch["mfcc_len"], batch["letters"]))
        CTC_loss = np.average(np.array([result['loss'] for result in results]))
        ED = np.average(np.array([result['edit_distance']
                                  for result in results]))
        return CTC_loss, ED

    @tf.function(experimental_relax_shapes=True)
    def predict_batch(self, mfcc, mfcc_lens):
        logits = self._compute_logits(mfcc, mfcc_lens, training=False)
        predicts = self._ctc_predict(logits, mfcc_lens)
        return self._to_dense(predicts)

    def predict(self, dataset, args):
        sentences = []
        for batch in dataset.batches(args.batch_size):
            for prediction, prediction_len in zip(*self.predict_batch(batch["mfcc"], batch["mfcc_len"])):
                sentences.append(prediction[:prediction_len])
        return sentences


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16,
                        type=int, help="Batch size.")
    parser.add_argument("--ctc_beam", default=10, type=int, help="CTC beam.")
    parser.add_argument("--epochs", default=20, type=int,
                        help="Number of epochs.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int,
                        help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False,
                        action="store_true", help="Verbose TF logging.")
    parser.add_argument("--rnn_cell_dim", default=512,
                        type=int, help="RNN cell dimension.")
    parser.add_argument("--clip_gradient", default=1.,
                        type=float, help="Norm for gradient clipping.")
    parser.add_argument("--brnn_layers", default=4, type=int,
                        help="Number of bidirectional RNN layers.")
    parser.add_argument("--lr", default=1e-3, type=float,
                        help="Learning rate.")
    parser.add_argument("--decay", default=True,
                        action="store_true", help="Learning rate decay.")
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
        ",".join(("{}={}".format(re.sub(
            "(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    timit = TimitMFCC()

    # Create the network and train
    network = Network(args)
    for epoch in range(args.epochs):
        metrics_train = network.train_epoch(timit.train, args)
        metrics_dev = network.evaluate(timit.dev, "dev", args)
        print('Epoch {}:'.format(epoch + 1))
        print('    train:\tCTC_loss={:.1f},\tedit_distance={:.3f}\n    dev:\tCTC_loss={:.1f},\tedit_distance={:.3f}'.format(
            metrics_train[0], metrics_train[1], metrics_dev[0], metrics_dev[1]))
    # Generate test set annotations, but to allow parallel execution, create it
    # in in args.logdir if it exists.
    out_path = "speech_recognition_test.txt"
    if os.path.isdir(args.logdir):
        out_path = os.path.join(args.logdir, out_path)
    with open(out_path, "w", encoding="utf-8") as out_file:
        for sentence in network.predict(timit.test, args):
            print(" ".join(timit.LETTERS[letters]
                           for letters in sentence), file=out_file)