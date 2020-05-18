#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from cags_dataset import CAGS
import efficient_net

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=15, type=int, help="Number of epochs.")
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

    def train_augment(image, label):
        if tf.random.uniform([]) >= 0.5:
            image = tf.image.flip_left_right(image)
        image = tf.image.resize_with_crop_or_pad(image, CAGS.H + 6, CAGS.W + 6)
        image = tf.image.resize(image, [tf.random.uniform([], minval=CAGS.H, maxval=CAGS.H + 12, dtype=tf.int32),
                                        tf.random.uniform([], minval=CAGS.W, maxval=CAGS.W + 12, dtype=tf.int32)])
        image = tf.image.random_crop(image, [CAGS.H, CAGS.W, CAGS.C])
        return image, label

    # Load the data
    cags = CAGS()

    train = cags.train.map(CAGS.parse)
    train = train.map(lambda example: (example["image"], example["label"]))
    train = train.shuffle(10000, seed=args.seed)
    train = train.map(train_augment)
    train = train.batch(args.batch_size)
    
    dev = cags.dev.map(CAGS.parse)
    # print(len(list(dev.as_numpy_iterator())))
    dev = dev.map(lambda example: (example["image"], example["label"]))
    dev = dev.batch(args.batch_size)

    test = cags.test.map(CAGS.parse)
    test = test.map(lambda example: example["image"])
    test = test.batch(args.batch_size)

    # Load the EfficientNet-B0 model
    efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False)
    
    # reg = tf.keras.regularizers.L1L2(0.0, 1e-3)
    base_model = efficientnet_b0    #Model with multiple outputs (stored in list called outputs)
    base_model.trainable = False
    classification_layer = tf.keras.layers.Dense(1024, activation='relu')(base_model.outputs[0])
    # classification_layer = tf.keras.layers.Dropout(0.8)(classification_layer) #0.6
    classification_layer = tf.keras.layers.Dense(1024, activation='relu')(classification_layer)
    # classification_layer = tf.keras.layers.Dropout(0.8)(classification_layer)
    outputs = tf.keras.layers.Dense(len(CAGS.LABELS), activation='softmax')(classification_layer)
    model = tf.keras.Model(inputs=efficientnet_b0.inputs, outputs=outputs)

    # model.summary()

    model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )    
    history = model.fit(train,
                        shuffle=False,
                        epochs=args.epochs,
                        validation_data=dev
    )

    model.trainable = True
    for layer in base_model.layers[:150]:
        layer.trainable =  False

    model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )    
    model.fit(
        train,
        shuffle=False,
        epochs=args.epochs + 20,
        initial_epoch =  history.epoch[-1],
        validation_data=dev,
    )
    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "cags_classification.txt"), "w", encoding="utf-8") as out_file:
        test_probabilities = model.predict(test, batch_size=args.batch_size)
        for probs in test_probabilities:
            print(np.argmax(probs), file=out_file)
