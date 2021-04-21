import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

"""Utilities methods"""


def file_path(relative_path):
    _dir = os.path.dirname(os.path.abspath(__file__))
    split_path = relative_path.split("/")
    new_path = os.path.join(_dir, *split_path)
    return new_path


def log(path, file):
    # check if the file exist
    log_file = os.path.join(path, file)

    if not os.path.isfile(log_file):
        open(log_file, "w+").close()

    console_logging_format = "%(message)s"
    file_logging_format = "%(message)s"

    # configure logger
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()

    # create a file handler for output file
    handler = logging.FileHandler(log_file)

    # set the logging level for log file
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger


def default_log():
    console_logging_format = "%(message)s"

    # configure logger
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()

    return logger


def tf_log(writer, name, value, step):
    with writer.as_default():
        tf.summary.scalar(name, value, step=step)
        writer.flush()


def tf_log_list(writer, name, values, steps):
    with writer.as_default():
        for i in range(len(steps)):
            tf.summary.scalar(name, values[i], step=steps[i])
        writer.flush()


def learning_rate_decay(epoch, lr):
    return lr * 0.97


def test_n_way(one_shot_tests, model):
    """
    TODO
    :param ds_list:
    :param model:
    :return:
    """
    num_tests = len(one_shot_tests)
    num_correct = 0

    for one_shot_test in one_shot_tests:
        preds_tensor = model.predict(one_shot_test)

        preds_argmax_idx = np.argmax(preds_tensor)

        if preds_argmax_idx == 0:
            num_correct += 1

    return num_correct/num_tests


class OneShotLearningAccuracyTestCallback(Callback):

    def __init__(self, val_ds_list):
        super(OneShotLearningAccuracyTestCallback, self).__init__()

        self.val_ds_list = val_ds_list

    def on_epoch_end(self, epoch, logs={}):
        # perform one-shot learning accuracy test
        oneshot_accuracy_score = test_n_way(self.val_ds_list, self.model)

        print(f"one_shot_accuracy: {oneshot_accuracy_score}")

        tf.summary.scalar('one_shot_accuracy', data=oneshot_accuracy_score, step=epoch)

        return oneshot_accuracy_score

    def __deepcopy__(self, memo):
        return OneShotLearningAccuracyTestCallback(self.val_ds_list)
