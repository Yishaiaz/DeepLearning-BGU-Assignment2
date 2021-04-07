import os
import math
from typing import List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.optimizers import Adam, Adamax, Adagrad
from enum import Enum


def get_matching_non_matching_pairs(txt_file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    assert txt_file_path.endswith('.txt')
    matching_pairs = {'name': [], 'n1': [], 'n2': []}
    non_matching_pairs = {'name1': [], 'n1': [], 'name2': [], 'n2': []}
    with open(txt_file_path, 'r') as pairs_file:
        single_line = pairs_file.readline()
        number_of_pairs = int(single_line)
        # skip the first line
        single_line = pairs_file.readline()
        # iterate through samples
        while single_line != '' and single_line != None:
            split_single_line = single_line.split('\t')
            assert len(split_single_line) == 3 or len(split_single_line) == 4
            # matching pairs - only 3 entries in line
            if len(split_single_line) == 3:
                matching_pairs['name'].append(split_single_line[0])
                matching_pairs['n1'].append(split_single_line[1])
                matching_pairs['n2'].append(split_single_line[2])
            # non-matching pairs - only 4 entries in line
            else:
                non_matching_pairs['name1'].append(split_single_line[0])
                non_matching_pairs['n1'].append(split_single_line[1])
                non_matching_pairs['name2'].append(split_single_line[2])
                non_matching_pairs['n2'].append(split_single_line[3])

            single_line = pairs_file.readline()

    return pd.DataFrame(matching_pairs), pd.DataFrame(non_matching_pairs), number_of_pairs


def get_single_image(name: str, image_idx: int, images_root_path: str = 'lfw2Data/lfw2', flatten_im: bool= False, show_im: bool=False, format: str='jpg') -> np.array:
    assert format in ['jpg', 'png', 'jpeg']

    image_name = '{name}_{id}.{format}'.format(name=name, id=str(image_idx).zfill(4), format=format)
    image_full_path = os.sep.join([images_root_path, name, image_name])
    single_image = cv2.imread(image_full_path)

    if show_im: #todo: remove before submission
        plt.imshow(single_image)
        plt.show()

    if flatten_im:
        return single_image.flatten()
    return single_image


class SiameseNeuralNetwork():
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
                 input_shape: Tuple[int],
                 layers: Tuple[Any] = __standard_layers,
                 learning_rate: float = 0.00006,
                 optimizer_loss='binary_crossentropy'):

        first_twin_network = Input(input_shape)
        second_twin_network = Input(input_shape)

        model = Sequential()

        # add all layers including the embedding
        for layer in layers:
            model.add(layer)

        first_embedding = model(first_twin_network)
        second_embedding = model(second_twin_network)

        # calculate the distance between twins with lambda layer
        lambda_layer = Lambda(lambda embedding_vectors: keras_backend.abs(embedding_vectors[0] - embedding_vectors[1]))
        twins_dist = lambda_layer((first_embedding, second_embedding))

        # define output layer, classifying whether it is the same person in the image
        layer_output = Dense(1, activation='sigmoid')(twins_dist)

        self.model = Model(inputs=(first_twin_network, second_twin_network), outputs=layer_output)
        optimizer = Adam(lr=learning_rate)
        self.model.compile(loss=optimizer_loss, optimizer=optimizer)


    def train_model(self):
        pass

    def predict(self):
        pass