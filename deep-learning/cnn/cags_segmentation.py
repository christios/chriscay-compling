#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from cags_dataset import CAGS
import efficient_net

class CAGSMaskIoU(tf.metrics.Mean):
    """CAGSMaskIoU computes IoU for CAGS dataset masks predicted by binary classification"""

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_mask = tf.reshape(tf.math.round(y_true) == 1, [-1, CAGS.H * CAGS.W])
        y_pred_mask = tf.reshape(tf.math.round(y_pred) == 1, [-1, CAGS.H * CAGS.W])

        intersection_mask = tf.math.logical_and(y_true_mask, y_pred_mask)
        union_mask = tf.math.logical_or(y_true_mask, y_pred_mask)

        intersection = tf.reduce_sum(tf.cast(intersection_mask, tf.float32), axis=1)
        union = tf.reduce_sum(tf.cast(union_mask, tf.float32), axis=1)

        iou = tf.where(union == 0, 1., intersection / union)
        return super().update_state(iou, sample_weight)

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
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

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    def train_augment(image, label):
        if tf.random.uniform([]) >= 0.5:
            image = tf.image.flip_left_right(image)
        if tf.random.uniform([]) >= 0.5:
            image = tf.image.rot90(image)
        image = tf.image.resize_with_crop_or_pad(image, CAGS.H + 25, CAGS.W + 25)
        image = tf.image.resize(image, [tf.random.uniform([], minval=CAGS.H, maxval=CAGS.H + 100, dtype=tf.int32),
                                        tf.random.uniform([], minval=CAGS.W, maxval=CAGS.W + 100, dtype=tf.int32)])
        image = tf.image.random_crop(image, [CAGS.H, CAGS.W, CAGS.C])
        return image, label

    # Load the data
    cags = CAGS()

    train = cags.train.map(CAGS.parse)
    train = train.map(lambda example: (example["image"], example["mask"]))
    train = train.shuffle(10000, seed=args.seed)
    # train = train.map(train_augment)
    train = train.batch(args.batch_size)

    dev = cags.dev.map(CAGS.parse)
    dev = dev.map(lambda example: (example["image"], example["mask"]))
    dev = dev.batch(args.batch_size)

    test = cags.test.map(CAGS.parse)
    test = test.map(lambda example: example["image"])
    test = test.batch(args.batch_size)
    
    count = 3
    models = []
    # model ensembling
    for n in range(count):

        # Load the EfficientNet-B0 model
        efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False)
        base_model = efficientnet_b0
        # base_model.trainable = False

        classification_layer = tf.keras.layers.Dense(50176)(base_model.outputs[0])
        outputs = tf.keras.layers.Reshape((224, 224, 1))(classification_layer)
        
        model = tf.keras.Model(inputs=efficientnet_b0.inputs, outputs=outputs)
        args.seed += 5
        tf.random.set_seed(args.seed)
        models.append(model)

        models[-1].compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.Huber(),
        metrics=[CAGSMaskIoU()],
        )    
        history = models[-1].fit(train,
                            shuffle=False,
                            epochs=args.epochs,
                            validation_data=dev
        )

        lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                    initial_learning_rate=1e-3,
                    decay_steps=2142/args.batch_size*20,
                    end_learning_rate=1e-6
        )

        models[-1].compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        loss=tf.keras.losses.Huber(),
        metrics=[CAGSMaskIoU()],
        ) 
        ep = 20 + args.epochs
        history = models[-1].fit(
            train,
            shuffle=False,
            epochs=ep,
            initial_epoch =  history.epoch[-1],
            validation_data=dev
        )

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "cags_segmentation.txt"), "w", encoding="utf-8") as out_file:

        y_pred = []
        for model in range(count):
            y_pred.append(models[model].predict(test, batch_size=args.batch_size))
        # model ensembling
        y_pred_avg = tf.keras.layers.average(y_pred)
        test_masks = y_pred_avg
        for mask in test_masks:
            zeros, ones, runs = 0, 0, []
            for pixel in np.reshape(mask >= 0.5, [-1]):
                if pixel:
                    if zeros or (not zeros and not ones):
                        runs.append(zeros)
                        zeros = 0
                    ones += 1
                else:
                    if ones:
                        runs.append(ones)
                        ones = 0
                    zeros += 1
            runs.append(zeros + ones)
            print(*runs, file=out_file)
