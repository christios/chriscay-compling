#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from mnist import MNIST

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--dropout", default=0, type=float, help="Dropout regularization.")
    parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
    parser.add_argument("--hidden_layers", default="500", type=str, help="Hidden layer configuration.")
    parser.add_argument("--l2", default=0, type=float, help="L2 regularization.")
    parser.add_argument("--label_smoothing", default=0, type=float, help="Label smoothing.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    args = parser.parse_args([] if "__file__" not in globals() else None)
    args.hidden_layers = [int(hidden_layer) for hidden_layer in args.hidden_layers.split(",") if hidden_layer]

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
    mnist = MNIST()

    # TODO: Implement L2 regularization.
    # If `args.l2` is nonzero, create a `tf.keras.regularizers.L1L2` regularizer
    # and use it for all kernels (but not biases) of all Dense layers.
    reg = None
    if args.l2 != 0:
        reg = tf.keras.regularizers.L1L2(0.0, args.l2)

    # TODO: Implement dropout.
    # Add a `tf.keras.layers.Dropout` with `args.dropout` rate after the Flatten
    # layer and after each Dense hidden layer (but not after the output Dense layer).

    # Create the model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=[MNIST.H, MNIST.W, MNIST.C]))
    model.add(tf.keras.layers.Dropout(args.dropout))
    for hidden_layer in args.hidden_layers:
        model.add(tf.keras.layers.Dense(hidden_layer, activation=tf.nn.relu,
            kernel_regularizer=reg)
        )
        model.add(tf.keras.layers.Dropout(args.dropout))
    model.add(tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax,
        kernel_regularizer=reg)
    )

    # TODO: Implement label smoothing.
    # Apply the given smoothing. You will need to change the
    # `SparseCategorical{Crossentropy,Accuracy}` to `Categorical{Crossentropy,Accuracy}`
    # because `label_smooting` is supported only by `CategoricalCrossentropy`.
    # That means you also need to modify the labels of all three datasets
    # (i.e., `mnist.{train,dev,test}.data["labels"]`) from indices of the gold class
    # to a full categorical distribution (you can use either NumPy or there is
    # a helper method also in the `tf.keras.utils`).

    gold_train = np.zeros((mnist.train.data["labels"].size, 10))
    gold_dev = np.zeros((mnist.dev.data["labels"].size, 10))
    gold_test = np.zeros((mnist.test.data["labels"].size, 10))
    for example in range(gold_train.shape[0]):
        gold_train[example][mnist.train.data["labels"][example]] = 1.0
    for example in range(gold_dev.shape[0]):
        gold_dev[example][mnist.dev.data["labels"][example]] = 1.0
    for example in range(gold_test.shape[0]):
        gold_test[example][mnist.test.data["labels"][example]] = 1.0
    mnist.train.data["labels"] = gold_train
    mnist.dev.data["labels"] = gold_dev
    mnist.test.data["labels"] = gold_test

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
        metrics=[tf.metrics.CategoricalAccuracy(name="accuracy")],
    )

    tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)
    model.fit(
        mnist.train.data["images"][:5000], mnist.train.data["labels"][:5000],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
        callbacks=[tb_callback],
    )

    test_logs = model.evaluate(
        mnist.test.data["images"], mnist.test.data["labels"], batch_size=args.batch_size,
    )
    tb_callback.on_epoch_end(1, {"val_test_" + metric: value for metric, value in zip(model.metrics_names, test_logs)})

    accuracy = test_logs[model.metrics_names.index("accuracy")]
    with open("mnist_regularization.out", "w") as out_file:
        print("{:.2f}".format(100 * accuracy), file=out_file)
