#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from cifar10 import CIFAR10

class Network:
    def __init__(self, args):

        inputs = tf.keras.layers.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C])

        def add_layer(layer_prev, layer):

            if layer[0] == 'C':
                return tf.keras.layers.Conv2D(filters=int(layer[1]),
                    kernel_size=int(layer[2]), strides=int(layer[3]),
                    padding=layer[4], activation='relu')(layer_prev)
            elif layer[0] == 'CB':
                layer_prev = tf.keras.layers.Conv2D(filters=int(layer[1]), kernel_size=int(layer[2]),
                    strides=int(layer[3]), padding=layer[4], use_bias=False)(layer_prev)
                layer_prev = tf.keras.layers.BatchNormalization()(layer_prev)
                return tf.keras.layers.Activation('relu')(layer_prev)
            elif layer[0] == 'M':
                return tf.keras.layers.MaxPool2D(pool_size=int(layer[1]), strides=int(layer[2]))(layer_prev)
            elif layer[0] == 'F':
                return tf.keras.layers.Flatten()(layer_prev)
            elif layer[0] == 'D':
                return tf.keras.layers.Dropout(float(layer[1]))(layer_prev)
            elif layer[0] == 'H':
                return tf.keras.layers.Dense(int(layer[1]), activation=tf.nn.relu)(layer_prev)
            
        layers_R = re.search(r'\[(.*)\]', args.cnn)
        if layers_R:
            layers_R = layers_R.group(1).split(',')
        layers = re.sub(r'\[.*\]', '', args.cnn).split(',')
        hidden = inputs
        for layer in layers:
            layer = layer.split('-')
            if layer[0] == 'R':
                input_residual = hidden
                for layer_R in layers_R:
                    hidden = add_layer(hidden, layer_R.split('-'))
                hidden = tf.add(hidden, input_residual)
            else:
                hidden = add_layer(hidden, layer)

        outputs = tf.keras.layers.Dense(CIFAR10.LABELS, activation=tf.nn.softmax)(hidden)

        self._model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        self._tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)

    @property
    def model(self):
        return self._model

    def train(self, dataset, args):
        self._model.fit(
            dataset.train.data["images"], dataset.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(dataset.dev.data["images"], dataset.dev.data["labels"]),
            callbacks=[self._tb_callback],
        )

    def test(self, dataset, args):
        test_logs = self._model.evaluate(dataset.test.data["images"], dataset.test.data["labels"], batch_size=args.batch_size)
        self._tb_callback.on_epoch_end(1, {"val_test_" + metric: value for metric, value in zip(self._model.metrics_names, test_logs)})
        return test_logs[self._model.metrics_names.index("accuracy")]

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=None, type=int, help="Batch size.")
    parser.add_argument("--cnn",
        default='C-32-3-1-same,C-64-3-1-same,M-3-2,C-128-3-1-same,M-3-2,C-128-3-1-same,M-3-2,F,H-128,H-256,H-512,H-1024', type=str, help="CNN architecture.")
    parser.add_argument("--epochs", default=None, type=int, help="Number of epochs.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
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

    # Load data
    cifar = CIFAR10()

    model = Network(args)
    model.train(cifar, args)
    # model.test(cifar, args)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as out_file:
        for probs in model.model.predict(cifar.test.data["images"], batch_size=args.batch_size):
            print(np.argmax(probs), file=out_file)
