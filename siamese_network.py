from functools import partial
from logging import Logger
from typing import Tuple, Callable

import tensorflow as tf
from kerastuner import HyperModel
from tensorflow.keras import backend as keras_backend
from tensorflow.python.keras import Input, Sequential, Model, regularizers
from tensorflow.python.keras.callbacks import LearningRateScheduler, EarlyStopping, CSVLogger
from tensorflow.python.keras.layers import Lambda, Dense, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop
from tensorflow.python.ops.init_ops_v2 import RandomNormal

from utils import default_log, learning_rate_decay, OneShotLearningAccuracyTestCallback


class SiameseNeuralNetwork:
    """
    This class is a Siamese neural network implementation based on paper
    'Siamese Neural Networks for One-shot Image Recognition'.
    """
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 learning_rate: float = 0.01,
                 batch_size: int = 32,
                 optimizer: str = 'sgd',
                 optimizer_loss: str = 'binary_crossentropy',
                 dense_layer_size: int = 4096,
                 enable_batch_normalization: bool = False,
                 l2_regularizer: float = None,
                 bias_initializer: str = None,
                 conv2D_kernel_initializer: str = None,
                 dense_kernel_initializer: str = None,
                 distance_metric: str = None,
                 dropout_rate: float = None,
                 enable_learning_rate_decay_scheduler: bool = False,
                 logger: Logger = None,
                 verbose: int = 0,
                 seed: int = None):

        if seed is not None:
            tf.random.set_seed(seed)

        self._seed = seed
        self.logger = default_log() if logger is None else logger
        self._verbose = verbose

        self._input_shape = input_shape
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._optimizer = optimizer
        self._optimizer_loss = optimizer_loss

        self._dense_layer_size = dense_layer_size
        self._enable_batch_normalization = enable_batch_normalization
        self._l2_regularizer = l2_regularizer
        self._bias_initializer = bias_initializer
        self._conv2D_kernel_initializer = conv2D_kernel_initializer
        self._dense_kernel_initializer = dense_kernel_initializer
        self._distance_metric = distance_metric
        self._dropout_rate = dropout_rate
        self._enable_learning_rate_decay_scheduler = enable_learning_rate_decay_scheduler

        # build model and compile
        self.model = self._build_model()
        self._compile()

        self._trained = False

    def _build_model(self) -> Model:
        """
        Build the underlying model based on given hyperparameters
        :return: model
        """
        input_shape = self._input_shape

        first_twin_network = Input(input_shape)
        second_twin_network = Input(input_shape)

        conv2D_kernel_initializer = self._conv2D_kernel_initializer
        if self._conv2D_kernel_initializer is not None and self._conv2D_kernel_initializer == "default":
            conv2D_kernel_initializer = RandomNormal(mean=0., stddev=10e-2)

        dense_kernel_initializer = self._dense_kernel_initializer
        if self._dense_kernel_initializer is not None and self._dense_kernel_initializer == "default":
            dense_kernel_initializer = RandomNormal(mean=0., stddev=2*10e-1)

        bias_initializer = self._bias_initializer
        if self._bias_initializer is not None and self._bias_initializer == "default":
            bias_initializer = RandomNormal(mean=0.5, stddev=10e-2)

        conv2d_layer = partial(Conv2D,
                               activation="relu",
                               kernel_initializer=conv2D_kernel_initializer,
                               bias_initializer=bias_initializer,
                               kernel_regularizer=regularizers.l2(self._l2_regularizer) if self._l2_regularizer is not None else None)

        dense_layer = partial(Dense,
                              activation='sigmoid',
                              kernel_initializer=dense_kernel_initializer,
                              bias_initializer=bias_initializer,
                              kernel_regularizer=regularizers.l2(self._l2_regularizer) if self._l2_regularizer is not None else None)

        def _construct_conv2d_layer(_model, filters, kernel_size):
            _model.add(conv2d_layer(filters, kernel_size))
            if self._enable_batch_normalization:
                _model.add(BatchNormalization())
            _model.add(MaxPooling2D())
            if self._dropout_rate is not None:
                _model.add(Dropout(rate=self._dropout_rate))

        model = Sequential()
        _construct_conv2d_layer(model, 64, (10, 10))
        _construct_conv2d_layer(model, 128, (7, 7))
        _construct_conv2d_layer(model, 128, (4, 4))
        _construct_conv2d_layer(model, 256, (4, 4))

        model.add(Flatten())

        model.add(dense_layer(self._dense_layer_size))

        first_embedding = model(first_twin_network)
        second_embedding = model(second_twin_network)

        # calculate the distance between twins with lambda layer
        if self._distance_metric is None or self._distance_metric == "abs":
            lambda_layer = Lambda(lambda embedding_vectors: keras_backend.abs(embedding_vectors[0] - embedding_vectors[1]))
        else:
            lambda_layer = Lambda(lambda embedding_vectors:
                                  keras_backend.sqrt(keras_backend.sum(keras_backend.square(embedding_vectors[0] - embedding_vectors[1]), axis=1, keepdims=True)))

        twins_dist = lambda_layer((first_embedding, second_embedding))

        # define output layer, classifying whether it is the same person in the image
        layer_output = dense_layer(1)(twins_dist)

        model = Model(inputs=(first_twin_network, second_twin_network), outputs=layer_output)

        if self._verbose > 2:
            self.logger.info("Model structure:\n{}\n".format(model.summary()))

        return model

    def _compile(self):
        """
        Compile the model with an optimizer
        :return: None
        """
        if self._optimizer == 'sgd':
            opt = SGD(learning_rate=self._learning_rate, momentum=0.5)
        elif self._optimizer == 'adam':
            opt = Adam(learning_rate=self._learning_rate)
        else:
            opt = RMSprop(learning_rate=self._learning_rate)

        self.model.compile(optimizer=opt, loss=self._optimizer_loss, metrics=[tf.keras.metrics.BinaryAccuracy()], run_eagerly=False)

    def summary(self) -> None:
        """
        Invokes the keras model summary() method
        :return: None
        """
        self.model.summary()

    def save(self, model_dir_path: str):
        """
        Invokes the keras model save() method
        :param model_dir_path: directory path instruct where to save the model
        :return: None
        """
        self.model.save(model_dir_path + "/model.h5")

    def load(self, model_dir_path: str):
        """
        Invkoes the keras model load_model() method
        :param model_dir_path: directory path instruct from where to load the model
        :return: None
        """
        self.model = load_model(model_dir_path + "/model.h5")
        self._trained = True

    def train(self,
              train_ds,
              val_ds,
              one_shot_val_ds_list,
              log_name,
              tf_writer,
              tf_log_dir,
              max_epoch_num: int = 50,
              patience: int = 20,
              learning_rate_decay_callback: Callable[[int, float], float] = lambda epoch, lr: learning_rate_decay(epoch, lr),
              seed: int = None):
        if self._trained:
            raise ValueError("Network was already trained")

        # handle seed
        if seed is not None:
            tf.random.set_seed(seed)

        # configure tensorboard callback and writer
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tf_log_dir)
        tf_writer.set_as_default()

        # configure early stopping callback
        early_stop_callback = EarlyStopping(monitor='val_binary_accuracy',
                                            min_delta=1,
                                            patience=patience,
                                            verbose=1,
                                            restore_best_weights=True)

        # configure one shot learning accuracy test callback
        one_shot_accuracy_test_callback = OneShotLearningAccuracyTestCallback(one_shot_val_ds_list)

        callbacks = [early_stop_callback, CSVLogger(log_name), one_shot_accuracy_test_callback, tensorboard_callback]

        if self._enable_learning_rate_decay_scheduler:
            callbacks.append(LearningRateScheduler(learning_rate_decay_callback))

        history = self.model.fit(train_ds, epochs=max_epoch_num, validation_data=val_ds, callbacks=callbacks)

        self._trained = True

        return history


class SiameseNeuralNetworkHyperModel(HyperModel):
    def __init__(self,
                 input_shape,
                 batch_size,
                 seed: int = None):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.seed = seed

    def build(self, hp):
        learning_rate = hp.get('learning_rate')
        dense_layer_size = hp.get('dense_layer_size')
        enable_batch_normalization = hp.get('enable_batch_normalization')
        bias_initializer = hp.get('bias_initializer')
        conv2D_kernel_initializer = hp.get('conv2D_kernel_initializer')
        dense_kernel_initializer = hp.get('dense_kernel_initializer')
        distance_metric = hp.get('distance_metric')
        dropout_rate = hp.get('dropout_rate')
        l2_regularizer = hp.get('l2_regularizer')
        optimizer = hp.get('optimizer')

        siamese_network = SiameseNeuralNetwork(self.input_shape,
                                               learning_rate=learning_rate,
                                               batch_size=self.batch_size,
                                               optimizer=optimizer,
                                               dense_layer_size=dense_layer_size,
                                               enable_batch_normalization=enable_batch_normalization,
                                               l2_regularizer=l2_regularizer if l2_regularizer != -1.0 else None,
                                               bias_initializer=bias_initializer,
                                               conv2D_kernel_initializer=conv2D_kernel_initializer,
                                               dense_kernel_initializer=dense_kernel_initializer,
                                               distance_metric=distance_metric,
                                               dropout_rate=dropout_rate if dropout_rate != 0.0 else None,
                                               seed=self.seed)

        return siamese_network.model
