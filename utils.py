import logging
import os
import tensorflow as tf


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
