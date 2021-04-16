import datetime
import gc
import os
import shutil

import numpy as np
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from config import HP_DENSE_UNITS_NUM, HP_BATCH_SIZE, HP_LEARNING_RATE, HP_OPTIMIZER
from preprocessing_utils import make_dataset
from siamese_network import SiameseNeuralNetwork
from utils import file_path, log

# Global configuration
seed = 0
lfw_a_input_shape = (250, 250, 1)
images_directory = 'lfw2Data/lfw2'
augment_dataset = True

# metrics
metric_training_binary_accuracy = "Training binary accuracy"
metric_validation_binary_accuracy = "Validation binary accuracy"
metric_test_binary_accuracy = "Test binary accuracy"
metric_training_loss = "Training loss"
metric_validation_loss = "Validation loss"
metric_test_loss = "Test loss"
metric_epoch_number = "# epochs"
hparams_metrics = [hp.Metric(metric_training_binary_accuracy, display_name=metric_training_binary_accuracy),
                   hp.Metric(metric_validation_binary_accuracy, display_name=metric_validation_binary_accuracy),
                   hp.Metric(metric_test_binary_accuracy, display_name=metric_test_binary_accuracy),
                   hp.Metric(metric_training_loss, display_name=metric_training_loss),
                   hp.Metric(metric_validation_loss, display_name=metric_validation_loss),
                   hp.Metric(metric_test_loss, display_name=metric_test_loss),
                   hp.Metric(metric_epoch_number, display_name=metric_epoch_number)]


tf.random.set_seed(seed)
np.random.seed(seed)


# def configure_hyper_param_grid():
#     HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['sgd']))
#     HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.1, 0.01, 0.001, 0.0001]))
#     HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32, 64, 128, 512]))
#     HP_DENSE_UNITS_NUM = hp.HParam('dense_units_num', hp.Discrete([4096, 2048, 1024]))
#
#     return HP_OPTIMIZER, HP_LEARNING_RATE, HP_BATCH_SIZE, HP_DENSE_UNITS_NUM


def run(train_ds,
        val_ds,
        test_ds,
        experiment_num,
        experiment_name,
        current_time,
        log_filename,
        input_shape,
        hparams,
        logger,
        patience: int = 20):

    print("\nRun experiment {}:\n{}\n".format(experiment_num, {h.name: hparams[h] for h in hparams}))

    tf_writer = tf.summary.create_file_writer(file_path("/tf_logs/") + f"hparam_tuning/{current_time}/{experiment_name}")

    with tf_writer.as_default():
        # log hyper-parameters into tensorboard
        hp.hparams(hparams)

        # initialize model
        siamese_network = SiameseNeuralNetwork(input_shape, batch_size=hparams[HP_BATCH_SIZE], seed=seed)

        # train on the given train/validation datasets
        training_history = siamese_network.train(train_ds, val_ds, log_filename, patience=patience, seed=seed, tf_writer=tf_writer)

        # extracts train/val statistics
        epoch = len(training_history.history['loss'])
        train_binary_accuracy = max(training_history.history['binary_accuracy'][-patience:]) if len(training_history.history['binary_accuracy']) > patience else training_history.history['binary_accuracy'][-1]
        validation_binary_accuracy = max(training_history.history['val_binary_accuracy'][-patience:]) if len(training_history.history['val_binary_accuracy']) > patience else training_history.history['val_binary_accuracy'][-1]
        train_loss = min(training_history.history['loss'][-patience:] )if len(training_history.history['loss']) > patience else training_history.history['loss'][-1]
        validation_loss = min(training_history.history['val_loss'][-patience:]) if len(training_history.history['val_loss']) > patience else training_history.history['val_loss'][-1]

        # evaluate on the test set
        test_loss, test_binary_accuracy = siamese_network.model.evaluate(test_ds)

        # summary
        logger.info("")
        logger.info("Summary:")
        logger.info("Training Epoch number [{}]".format(epoch))
        logger.info("Final LR [{}]".format(training_history.history['lr']))
        logger.info("Training binary accuracy: [{}]".format(train_binary_accuracy))
        logger.info("Validation binary accuracy: [{}]".format(validation_binary_accuracy))
        logger.info("Test binary accuracy: [{}]".format(test_binary_accuracy))
        logger.info("Training loss: [{}]".format(train_loss))
        logger.info("Validation loss: [{}]".format(validation_loss))
        logger.info("Test loss: [{}]".format(test_loss))

        # log into hparams tensorboard
        tf.summary.scalar(metric_training_binary_accuracy, train_binary_accuracy, step=experiment_num)
        tf.summary.scalar(metric_validation_binary_accuracy, validation_binary_accuracy, step=experiment_num)
        tf.summary.scalar(metric_test_binary_accuracy, test_binary_accuracy, step=experiment_num)
        tf.summary.scalar(metric_training_loss, train_loss, step=experiment_num)
        tf.summary.scalar(metric_validation_loss, validation_loss, step=experiment_num)
        tf.summary.scalar(metric_test_loss, test_loss, step=experiment_num)
        tf.summary.scalar(metric_epoch_number, epoch, step=experiment_num)

        # save model
        model_dir_name = f"{current_time}_{experiment_num}_{experiment_name}"
        os.mkdir(file_path("/models/") + model_dir_name)
        siamese_network.save(file_path(f"/models/{model_dir_name}"))

        logger.info("\n#################################################################")


def run_experiments():
    # remove previous tensor-board events files
    tf_logs_dir = file_path("/tf_logs/")
    if os.path.isdir(tf_logs_dir):
        shutil.rmtree(tf_logs_dir)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    with tf.summary.create_file_writer(tf_logs_dir + "/hparam_tuning").as_default():
        hp.hparams_config(
            hparams=[HP_OPTIMIZER, HP_LEARNING_RATE, HP_BATCH_SIZE, HP_DENSE_UNITS_NUM],
            metrics=hparams_metrics,
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
                                                             augment_training_dataset=augment_dataset,
                                                             seed=seed)

                    run(train_ds,
                        val_ds,
                        test_ds,
                        experiment_num,
                        experiment_name,
                        current_time,
                        log_filename,
                        lfw_a_input_shape,
                        hparams,
                        logger)

                    # collect garbage to save memory
                    gc.collect()


if __name__ == "__main__":
    run_experiments()
