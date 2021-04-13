import datetime
import gc
import os
import shutil

import numpy as np
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from preprocessing_utils import make_dataset
from siamese_network import SiameseNeuralNetwork
from utils import file_path, log


# Global configuration
seed = 0
lfw_a_input_shape = (250, 250, 1)
images_directory = 'lfw2Data/lfw2'

tf.random.set_seed(seed)
np.random.seed(seed)


def configure_hyper_param_grid():
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['sgd']))
    HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.1, 0.01, 0.001, 0.0001]))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32, 64, 128, 512]))
    HP_DENSE_UNITS_NUM = hp.HParam('dense_units_num', hp.Discrete([4096, 2048, 1024]))

    return HP_OPTIMIZER, HP_LEARNING_RATE, HP_BATCH_SIZE, HP_DENSE_UNITS_NUM


def run(train_ds,
        val_ds,
        test_ds,
        experiment_num,
        experiment_name,
        current_time,
        input_shape,
        hparams,
        batch_size,
        optimizer,
        learning_rate,
        dense_units_num,
        logger):

    logger.info("\nRun experiment {}:\n{}\n".format(experiment_num, {h.name: hparams[h] for h in hparams}))

    tf_writer = tf.summary.create_file_writer(file_path("/tf-logs/") + f"hparam_tuning/{current_time}/{experiment_num}/{experiment_name}")

    with tf_writer.as_default():
        hp.hparams(hparams)

        siamese_network = SiameseNeuralNetwork(input_shape, seed=seed, batch_size=batch_size)

        siamese_network.train(train_ds, val_ds)

        # save model
        model_dir_name = f"{current_time}_{experiment_num}_{experiment_name}"
        os.mkdir(file_path("/models/") + model_dir_name)
        siamese_network.save(file_path(f"/models/{model_dir_name}"))

        logger.info("\n#################################################################")


# def data_loading_and_preprocessing():
#     #train_matching_pairs, train_non_matching_pairs, number_of_pairs = get_matching_non_matching_pairs(file_path("/lfw2Data/pairsDevTrain.txt"))
#     #train_matching_pairs, train_non_matching_pairs = load_images_as_vectors(train_matching_pairs, train_non_matching_pairs)
#
#     # test_matching_pairs, test_non_matching_pairs = get_matching_non_matching_pairs("/lfw2Data/pairsDevTest.txt")
#     # test_ds, y_test = load_images_as_vectors(test_matching_pairs, test_non_matching_pairs)
#
#     # X_train_matching_pairs, X_val_matching_pairs, y_train_matching_pairs, y_val_matching_pairs = train_test_split(train_matching_pairs, np.ones(len(train_matching_pairs)), test_size=0.2, shuffle=True, random_state=seed)
#     # X_train_non_matching_pairs, X_val_non_matching_pairs, y_train_non_matching_pairs, y_val_non_matching_pairs = train_test_split(train_non_matching_pairs, np.zeros(len(train_non_matching_pairs)), test_size=0.2, shuffle=True, random_state=seed)
#     #
#     # train_ds_match = tf.data.Dataset.from_generator(
#     #     load_images_as_vectors_generator,
#     #     output_types=(tf.Tensor, tf.Tensor),
#     #     #output_shapes=(X_train_matching_pairs.shape[0], 2, 250, 250, 1),
#     #     args=X_train_matching_pairs)
#     # train_ds_non_match = tf.data.Dataset.from_generator(
#     #     load_images_as_vectors_generator,
#     #     output_types=(tf.Tensor, tf.Tensor),
#     #     #output_shapes=(X_train_non_matching_pairs.shape[0], 2, 250, 250, 1),
#     #     args=X_train_non_matching_pairs)
#     #
#     # train_ds = tf.data.Dataset.zip((train_ds_match, train_ds_non_match)).batch(32).prefetch(tf.data.AUTOTUNE).cache()
#
#     # X_train = np.concatenate([X_train_matching_pairs, X_train_non_matching_pairs])
#     # X_val = np.concatenate([X_val_matching_pairs, X_val_non_matching_pairs])
#     #
#     # train_ds = tf.data.Dataset.from_generator(
#     #     load_images_as_vectors_generator,
#     #     output_types=(tf.Tensor, tf.Tensor),
#     #     output_shapes=(X_train.shape[0], 2, 250, 250, 1),
#     #     args=X_train)
#     # y_train = np.concatenate([y_train_matching_pairs, y_train_non_matching_pairs])
#     # y_val = np.concatenate([y_val_matching_pairs, y_val_non_matching_pairs])
#     train_ds, val_ds, test_ds = make_dataset("", 32, seed)
#     #return X_train, X_val, y_train, y_val
#     return train_ds, val_ds


def run_experiments():
    # remove previous tensor-board events files
    tf_logs_dir = file_path("/tf_logs/")
    if os.path.isdir(tf_logs_dir):
        shutil.rmtree(tf_logs_dir)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # configure hyper-parameters grid-search
    HP_OPTIMIZER, HP_LEARNING_RATE, HP_BATCH_SIZE, HP_DENSE_UNITS_NUM = configure_hyper_param_grid()

    # metrics
    metric_oneshot_accuracy = "validation oneshot_accuracy"

    with tf.summary.create_file_writer(tf_logs_dir + "/hparam_tuning").as_default():
        hp.hparams_config(
            hparams=[HP_OPTIMIZER, HP_LEARNING_RATE, HP_BATCH_SIZE, HP_DENSE_UNITS_NUM],
            metrics=[hp.Metric(metric_oneshot_accuracy, display_name=metric_oneshot_accuracy)],
        )

    experiment_num = 0

    for dense_units_num in HP_DENSE_UNITS_NUM.domain.values:
        for batch_size in HP_BATCH_SIZE.domain.values:
            for learning_rate in HP_LEARNING_RATE.domain.values:
                for optimizer in HP_OPTIMIZER.domain.values:
                    hparams = {
                        HP_OPTIMIZER: optimizer,
                        HP_LEARNING_RATE: learning_rate,
                        HP_BATCH_SIZE: batch_size,
                        HP_DENSE_UNITS_NUM: dense_units_num,
                    }

                    experiment_num += 1
                    experiment_name = "batch_size={}," \
                                      "optimizer={}," \
                                      "learning_rate={}," \
                                      "dense_units_num={}".format(batch_size,
                                                                  optimizer,
                                                                  learning_rate,
                                                                  dense_units_num)

                    # init experiments logger
                    log_filename = file_path(f"experiments/{current_time}_{experiment_num}_{experiment_name}.txt")

                    if os.path.exists(log_filename):
                        os.remove(log_filename)

                    logger = log(".", log_filename)
                    logger.info("#################################################################")

                    # preprocessing and dataset creation
                    train_ds, val_ds, test_ds = make_dataset(images_directory=images_directory,
                                                             batch_size=batch_size,
                                                             seed=seed)
                    # YOUR VERSION
                    # train_ds, val_ds, test_ds = make_dataset(batch_size=batch_size,
                    #                                          seed=seed)

                    run(train_ds,
                        val_ds,
                        test_ds,
                        experiment_num,
                        experiment_name,
                        current_time,
                        lfw_a_input_shape,
                        hparams,
                        batch_size,
                        optimizer,
                        learning_rate,
                        dense_units_num,
                        logger)

                    # collect garbage to save memory
                    gc.collect()


if __name__ == "__main__":
    run_experiments()
