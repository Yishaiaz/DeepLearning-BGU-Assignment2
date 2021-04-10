from functools import partial
from logging import Logger
from typing import Tuple

from tensorflow.python.keras import Input, Sequential, Model, regularizers
from tensorflow.python.keras.layers import Lambda, Dense, Conv2D, BatchNormalization
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop
from tensorflow.keras import backend as keras_backend

from utils import default_log


class SiameseNeuralNetwork():
    """
    This class is a Siamese neural network implementation based on # TODO
    """
    __standard_layers = (Conv2D(64, (10, 10), activation='relu'),
                         MaxPooling2D(),
                         Conv2D(128, (7, 7), activation='relu'),
                         MaxPooling2D(),
                         Conv2D(128, (4, 4), activation='relu'),
                         MaxPooling2D(),
                         Conv2D(256, (4, 4), activation='relu'),
                         Flatten(),
                         Dense(4096, activation='sigmoid'))

    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 layers: Tuple[Any] = __standard_layers,
                 learning_rate: float = 0.00006,
                 batch_size: int = 32,
                 optimizer: str = 'sgd',
                 optimizer_loss: str = 'binary_crossentropy',
                 logger: Logger = None,
                 verbose: int = 0):

        self.logger = default_log() if logger is None else logger
        self._verbose = verbose

        self._input_shape = input_shape
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._optimizer = optimizer
        self._optimizer_loss = optimizer_loss

        self._seed = None

        # build model
        self._model = self._build_model()
        self._compile(self._model)

        self._trained = False


    def _build_model(self):
        input_shape = self._input_shape

        first_twin_network = Input(input_shape)
        second_twin_network = Input(input_shape)

        RegularizedConv2D = partial(Conv2D,
                                    activation="relu",
                                    kernel_initializer="glorot_normal",
                                    kernel_regularizer=regularizers.l2(0.01))

        model = Sequential()
        model.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape))
        if self._enable_batch_normalization:
            model.add(BatchNormalization())
        model.add(model.MaxPooling2D())

        # Conv2D(128, (7, 7), activation='relu'),
        # MaxPooling2D(),
        # Conv2D(128, (4, 4), activation='relu'),
        # MaxPooling2D(),
        # Conv2D(256, (4, 4), activation='relu'),
        # Flatten(),
        # Dense(4096, activation='sigmoid')
        # # add all layers including the embedding
        # for layer in layers:
        #     model.add(layer)

        first_embedding = model(first_twin_network)
        second_embedding = model(second_twin_network)

        # calculate the distance between twins with lambda layer
        lambda_layer = Lambda(lambda embedding_vectors: keras_backend.abs(embedding_vectors[0] - embedding_vectors[1]))
        twins_dist = lambda_layer((first_embedding, second_embedding))

        # define output layer, classifying whether it is the same person in the image
        layer_output = Dense(1, activation='sigmoid')(twins_dist)

        model = Model(inputs=(first_twin_network, second_twin_network), outputs=layer_output)

        if self._verbose > 2:
            self.logger.info("Model structure:\n{}\n".format(model.summary()))

        return model

    def _compile(self):
        if self._optimizer == 'sgd':
            opt = SGD(learning_rate=self._learning_rate)
        elif self._optimizer == 'adam':
            opt = Adam(learning_rate=self._learning_rate)
        else:
            opt = RMSprop(learning_rate=self._learning_rate)

        self._model.compile(optimizer=opt, loss=self._optimizer_loss)

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
        self._model.save(model_dir_path + "/model.h5")


    def load(self, model_dir_path: str):
        """
        Invkoes the keras model load_model() method
        :param model_dir_path: directory path instruct from where to load the model
        :return: None
        """
        self._model = load_model(model_dir_path + "/model.h5")
        self._trained = True

    def train(self):
        if self._trained:
            raise ValueError("Network was already trained")

        # handle seed TODO

        self._trained = True

    def predict(self):
        if not self._trained:
            raise ValueError("Network wasn't trained")

        pass
