#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from mnist import MNIST

class Model(tf.Module):
    def __init__(self, args):

        self._W1 = tf.Variable(tf.random.normal([MNIST.W * MNIST.H * MNIST.C, args.hidden_layer], stddev=0.1, seed=args.seed), trainable=True)
        self._b1 = tf.Variable(tf.zeros([args.hidden_layer]), trainable=True)
        self._W2 = tf.Variable(tf.random.normal([args.hidden_layer, MNIST.LABELS], stddev=0.1, seed=args.seed), trainable=True)
        self._b2 = tf.Variable(tf.zeros([MNIST.LABELS]), trainable=True)

    def predict(self, inputs):

        inputs = tf.convert_to_tensor(inputs["images"])
        inputs = tf.reshape(inputs, [inputs.shape[0], -1])
        Z1 = tf.linalg.matmul(inputs, self._W1) + self._b1
        A1 = tf.nn.tanh(Z1)
        Z2 = tf.linalg.matmul(A1, self._W2) + self._b2
        A2 = tf.nn.softmax(Z2)

        return inputs, A1, A2, Z1

    def train_epoch(self, dataset):
        for batch in dataset.batches(args.batch_size):

            # Compute the input layer, hidden layer and output layer
            NN = self.predict(batch)

            # Compute the loss:
            # - for every batch example, it is the categorical crossentropy of the
            #   predicted probabilities and gold batch label
            # - finally, compute the average across the batch examples
            gold = np.zeros((batch["labels"].size, 10))
            for example in range(gold.shape[0]):
                gold[example][batch["labels"][example]] = 1.0
            loss = tf.math.reduce_mean(tf.keras.losses.categorical_crossentropy(tf.convert_to_tensor(gold), NN[2]))

            # TODO: Compute the gradient of the loss with respect to all
            # variables.
            dZ2 = NN[2] - tf.convert_to_tensor(gold, tf.float32)        # foo and dfoo have always the same dimensions
            d_W2 = tf.reduce_mean(tf.einsum("ai,aj->aji", dZ2, NN[1]), axis=0)
            d_b2 = tf.reduce_mean(dZ2, axis=0)      # This is m x 1 and not m x h_out because of boradcasting (when computing Z1 in predict())
            dZ1 = tf.math.multiply(tf.linalg.matmul(dZ2, self._W2, transpose_b=True), tf.math.square(tf.math.reciprocal(tf.math.cosh(NN[3]))))
            d_W1 = tf.reduce_mean(tf.einsum("ai,aj->aji", dZ1, NN[0]), axis=0)
            d_b1 = tf.reduce_mean(dZ1, axis=0)

            variables = [self._W1, self._b1, self._W2, self._b2]
            gradients = [d_W1, d_b1, d_W2, d_b2]

            #  SGD update with learning rate `args.learning_rate`
            # for the variable and computed gradient.
            for variable, gradient in zip(variables, gradients):
                variable.assign_sub(args.learning_rate * gradient)

    def evaluate(self, dataset):
        # Compute the accuracy of the model prediction
        correct = 0
        for batch in dataset.batches(args.batch_size):
            # TODO(sgd_backpropagation): Compute the probabilities of the batch images
            probabilities = self.predict(batch)[2]
            # TODO(sgd_backpropagation): Evaluate how many batch examples were predicted
            # correctly and increase `correct` variable accordingly.
            correct += sum(1 for i in range(batch["labels"].size) if tf.math.argmax(probabilities, axis=1).numpy()[i] == batch["labels"][i])
        return correct / dataset.size


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--hidden_layer", default=100, type=int, help="Size of the hidden layer.")
    parser.add_argument("--learning_rate", default=0.2, type=float, help="Learning rate.")
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
    mnist = MNIST()

    # Create the TensorBoard writer
    writer = tf.summary.create_file_writer(args.logdir, flush_millis=10*1000)

    # Create the model
    model = Model(args)

    for epoch in range(args.epochs):
        model.train_epoch(mnist.train)
        accuracy = model.evaluate(mnist.dev)

        print("Dev accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * accuracy), flush=True)
        with writer.as_default():
            tf.summary.scalar("dev/accuracy", 100 * accuracy, step=epoch + 1)

    accuracy = model.evaluate(mnist.test)
    print("Test accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * accuracy), flush=True)
    with writer.as_default():
        tf.summary.scalar("test/accuracy", 100 * accuracy, step=epoch + 1)

    # Save the test accuracy in percents rounded to two decimal places.
    with open("sgd_manual.out", "w") as out_file:
        print("{:.2f}".format(100 * accuracy), file=out_file)
