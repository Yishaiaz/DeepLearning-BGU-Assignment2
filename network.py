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
    """
    Reads the text files containing the matching and non matching images names and indexes
    The file path must end with '.txt'.
    The method assumes matching pairs lines are structured as <name>\t<image_idx>\t<image_idx>
    The method assumes non-matching pairs lines are structured as <name1>\t<image_idx>\t<name2>\t<image_idx>
    returns data frame object for both types (converts the index to int)
    :param txt_file_path: str, the path to the txt file
    :return: Tuple[matching pairs:pd.DataFrame, non-matching pairs: pd.DataFrame, int]
    """
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
            split_single_line = single_line.replace('\n', '').split('\t')
            assert len(split_single_line) == 3 or len(split_single_line) == 4
            # matching pairs - only 3 entries in line
            if len(split_single_line) == 3:
                matching_pairs['name'].append(split_single_line[0])
                matching_pairs['n1'].append(int(split_single_line[1]))
                matching_pairs['n2'].append(int(split_single_line[2]))
            # non-matching pairs - only 4 entries in line
            else:
                non_matching_pairs['name1'].append(split_single_line[0])
                non_matching_pairs['n1'].append(int(split_single_line[1]))
                non_matching_pairs['name2'].append(split_single_line[2])
                non_matching_pairs['n2'].append(int(split_single_line[3]))

            single_line = pairs_file.readline()

    return pd.DataFrame(matching_pairs), pd.DataFrame(non_matching_pairs), number_of_pairs


def get_single_image(name: str,
                     image_idx: int,
                     images_root_path: str = 'lfw2Data/lfw2',
                     flatten_im: bool= False,
                     show_im: bool=False,
                     im_format: str= 'jpg') -> np.array:
    """
    reads a single image file from the given images root directory.
    the function assumes each image is within a directory by the name of the person.
    the function assumes each image filename is <name>_####.<im_format>.
    the funciton supports the following formats ['jpg', 'png', 'jpeg'] and
    builds upon CV2.imread() method.
    if needed, the function returns the flatten image np.array (default is not to flatten)

    :param name: str, name of the person as it appears in the dataset
    :param image_idx: int, the image index to read
    :param images_root_path: the root folder for all images directories.
    :param flatten_im: bool, whether to flatten the image np.array
    :param show_im: bool, for visualization of the image at runtime.
    :param im_format: str, format of the image file
    :return: np.array, the image.
    """
    assert im_format in ['jpg', 'png', 'jpeg']

    image_name = '{name}_{id}.{format}'.format(name=name, id=str(image_idx).zfill(4), format=im_format)
    image_full_path = os.sep.join([images_root_path, name, image_name])
    single_image = cv2.imread(image_full_path)

    if show_im: #todo: remove before submission
        plt.imshow(single_image)
        plt.show()

    if flatten_im:
        return single_image.flatten()
    return single_image


def load_images_as_vectors(matching_df: pd.DataFrame,
                           non_matching_df: pd.DataFrame) -> Tuple[np.array, np.array]:
    """
    :param matching_df:
    :param non_matching_df:
    :return:
    """
    assert matching_df.shape[1] == 3 or matching_df.shape[1] == 0
    assert non_matching_df.shape[1] == 4 or non_matching_df.shape[1] == 0

    all_images = []
    all_labels = []
    # get all matching images
    label = 1
    for idx, row in matching_df.iterrows():
        person_name = row[0]
        first_image_idx = row[1]
        second_image_idx = row[2]
        im1 = get_single_image(person_name, first_image_idx)
        im2 = get_single_image(person_name, second_image_idx)
        all_images.append((im1, im2))
        all_labels.append(label)

    label = 0
    for idx, row in non_matching_df.iterrows():
        first_person_name = row[0]
        first_person_image_idx = row[1]
        second_person_name = row[2]
        second_person_image_idx = row[3]
        im1 = get_single_image(first_person_name, first_person_image_idx)
        im2 = get_single_image(second_person_name, second_person_image_idx)
        all_images.append((im1, im2))
        all_labels.append(label)

    return np.array(all_images), np.array(all_labels)


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
                 input_shape: Tuple[int, int, int],
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

    def summary(self) -> None:
        """
        invokes the keras model summary() method
        :return: None
        """
        self.model.summary()

    def train_model(self):
        pass

    def predict(self):
        pass


if __name__ == '__main__':
    snn = SiameseNeuralNetwork(input_shape=(110, 110, 3))
    snn.summary()

