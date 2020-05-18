#!/usr/bin/env python3
import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf

from mnist import MNIST

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", default=5, type=int, help="MNIST examples to use.")
    parser.add_argument("--iterations", default=10, type=int, help="Iterations of the power algorithm.")
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

    # Load data
    mnist = MNIST()

    data = tf.convert_to_tensor(mnist.train.data["images"][:args.examples])
    data = tf.reshape(data, [data.shape[0], data.shape[1] * data.shape[2] * data.shape[3]])
    mean = tf.math.reduce_mean(data, axis=0)

    cov = tf.linalg.matmul(data - mean, data - mean, transpose_a=True)

    total_variance = tf.math.reduce_sum(tf.linalg.diag_part(cov))

    v = tf.ones(cov.shape[0], dtype=tf.float32)
    for i in range(args.iterations):
        v = tf.linalg.matvec(cov,v)
        s = tf.linalg.norm(v)
        v /= s

    # The `v` is now the eigenvector of the largest eigenvalue, `s`. We now
    # compute the explained variance, which is a ration of `s` and `total_variance`.
    explained_variance = s / total_variance

    with open("pca_first.out", "w") as out_file:
        print("{:.2f} {:.2f}".format(total_variance / args.examples, 100 * explained_variance), file=out_file)