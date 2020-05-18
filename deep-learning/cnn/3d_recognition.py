#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from modelnet import ModelNet

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--modelnet", default=20, type=int, help="ModelNet dimension.")
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

    # Load the data
    modelnet = ModelNet(args.modelnet)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(
        (args.modelnet, args.modelnet, args.modelnet, 1)))
    model.add(tf.keras.layers.Conv3D(16, kernel_size=3,
                                          strides=1, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling3D(
        pool_size=2, strides=2, padding='valid'))
    model.add(tf.keras.layers.Conv3D(32, kernel_size=3,
                                          strides=1, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling3D(
        pool_size=2, strides=2, padding='valid'))
    model.add(tf.keras.layers.Conv3D(128, kernel_size=3,
                                          strides=1, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling3D(
        pool_size=2, strides=2, padding='valid'))
    model.add(tf.keras.layers.Conv3D(256, kernel_size=3,
                                          strides=1, activation='relu', padding='same'))
    model.add(tf.keras.layers.GlobalAveragePooling3D())
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
                       optimizer=tf.optimizers.Adam(),
                       metrics=['accuracy'])
    model.fit(
        modelnet.train.data['voxels'], modelnet.train.data['labels'],
        batch_size=args.batch_size,
        steps_per_epoch=int(modelnet.train.size/args.batch_size),
        epochs=args.epochs,
        validation_data=(
            modelnet.dev.data['voxels'], modelnet.dev.data['labels']),
        shuffle=True
    )
    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "3d_recognition.txt"), "w", encoding="utf-8") as out_file:
        test_probabilities = model.predict(
            modelnet.test.data['voxels'], batch_size=args.batch_size)
        for probs in test_probabilities:
            print(np.argmax(probs), file=out_file)
