import os
import random
from typing import Tuple, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from utils import file_path


def get_matching_non_matching_pairs(txt_file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Reads the text files containing the matching and non matching images names and indexes
    The file path must end with '.txt'.
    The method assumes matching pairs lines are structured as <name>\t<image_idx>\t<image_idx>
    The method assumes non-matching pairs lines are structured as <name1>\t<image_idx>\t<name2>\t<image_idx>
    returns data frame object for both types (converts the index to int)
    :param txt_file_path: str, the path to the txt file
    :return: Tuple[matching pairs:pd.DataFrame, non-matching pairs: pd.DataFrame, number of matching/non-matching pairs: int]
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
        while single_line != '' and single_line is not None:
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
                     flatten_im: bool = False,
                     show_im: bool = False,
                     normalize_im: bool = True,
                     resize_im: bool = False,
                     im_format: str = 'jpg') -> np.array:
    """
    Reads a single image file from the given images root directory.
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
    :param normalize_im: bool, whether to normalize the gray scale value by 255.
    :param resize_im: bool, whether to resize the image to a 250 by 250 pixels image.
    :param im_format: str, format of the image file
    :return: np.array, the image.
    """
    assert im_format in ['jpg', 'png', 'jpeg']

    image_name = '{name}_{id}.{format}'.format(name=name, id=str(image_idx).zfill(4), format=im_format)
    image_full_path = os.sep.join([images_root_path, name, image_name])
    single_image = cv2.imread(image_full_path, cv2.IMREAD_GRAYSCALE)

    if show_im:  # TODO: remove before submission
        plt.imshow(single_image)
        plt.show()

    if flatten_im:
        single_image = single_image.flatten()

    if normalize_im:
        single_image = single_image / 255

    if resize_im:
        single_image = cv2.resize(single_image, (250, 250))

    return single_image


def load_images_as_vectors(matching_df: pd.DataFrame,
                           non_matching_df: pd.DataFrame) -> Tuple[np.array, np.array, np.array]:
    """
    :param matching_df:
    :param non_matching_df:
    :return:
    """
    assert matching_df.shape[1] == 3 or matching_df.shape[1] == 0
    assert non_matching_df.shape[1] == 4 or non_matching_df.shape[1] == 0

    all_images = []
    all_labels = []
    all_names = []

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
        all_names.append((person_name, person_name))

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
        all_names.append((first_person_name, second_person_name))

    return np.array(all_images), np.array(all_labels), np.array(all_names)


def create_tf_dataset(image_pairs: np.array,
                      pairs_labels: np.array,
                      batch_size: int = 32) -> Tuple[tf.data.Dataset, tf.data.Dataset]:

    first_images = np.copy(image_pairs[:, 0])
    second_images = np.copy(image_pairs[:, 1])
    dataset = tf.data.Dataset.from_tensor_slices((first_images, second_images))
    labels = tf.data.Dataset.from_tensor_slices(pairs_labels)
    dataset = tf.data.Dataset.zip((dataset, labels))
    batched_dataset = dataset.batch(batch_size=batch_size).repeat() # to allow for multiple  epochs
    return dataset, batched_dataset


def train_val_split(image_pairs: np.array,
                    pairs_labels: np.array,
                    pairs_names: np.array,
                    val_percent: float = 0.2,
                    shuffle: bool = True) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array]:

    assert image_pairs.shape[0] == pairs_labels.shape[0]

    # collecting a list of indices where the label is 1 (matching)
    samples_for_val = list()
    number_of_samples_for_val = int(image_pairs.shape[0]*val_percent)
    while len(samples_for_val) < number_of_samples_for_val:
        # pick a random matching image pair
        random_idx = int(random.random()*len(image_pairs))
        if random_idx not in samples_for_val and pairs_labels[random_idx] == 1:
            samples_for_val.append(random_idx)

    # converting the list of indices to a boolean mask
    samples_for_val_mask = np.zeros(image_pairs.shape[0])
    samples_for_val_mask[samples_for_val] = 1
    samples_for_val_mask = np.array(samples_for_val_mask, dtype=bool)

    # splitting the original dataset into training and validation
    samples_for_val = np.copy(image_pairs[samples_for_val_mask])
    samples_for_train = np.copy(image_pairs[~samples_for_val_mask])
    labels_for_val = np.copy(pairs_labels[samples_for_val_mask])
    labels_for_train = np.copy(pairs_labels[~samples_for_val_mask])
    names_in_train = np.copy(pairs_names[~samples_for_val_mask])
    names_in_val = np.copy(pairs_names[samples_for_val_mask])

    if shuffle:
        # shuffles training and validation, maintains order across names labels and images.
        c = list(zip(samples_for_val, labels_for_val, names_in_val))
        random.shuffle(c)
        samples_for_val, labels_for_val, names_in_val = zip(*c)
        samples_for_val, labels_for_val, names_in_val = np.array(samples_for_val),\
                                                        np.array(labels_for_val),\
                                                        np.array(names_in_val)

        c = list(zip(samples_for_train, labels_for_train, names_in_train))
        np.random.shuffle(c)
        samples_for_train, labels_for_train, names_in_train = zip(*c)
        samples_for_train, labels_for_train, names_in_train = np.array(samples_for_train),\
                                                              np.array(labels_for_train),\
                                                              np.array(names_in_train)

    # creating a tensorflow dataset, both batched and non-batched for each set.
    train_dataset, train_batched_dataset = create_tf_dataset(samples_for_train, labels_for_train)
    val_dataset, val_batched_dataset = create_tf_dataset(samples_for_val, labels_for_val)
    return train_dataset, train_batched_dataset, names_in_train, val_dataset, val_batched_dataset, names_in_val

def create_nway_set(N: int,
                    pair_to_test_vector: np.array,
                    pair_to_test_name: str,
                    all_training_vectors: np.array,
                    all_names: List) -> np.array:

    im1_to_test = pair_to_test_vector[0]
    im2_to_test = pair_to_test_vector[1]
    n_idx = 0
    while n_idx < N-1:
        random_idx = int(random.random()*len(all_training_vectors))


def load_images_as_tensors(img_pairs_by_indices: pd.DataFrame):
    """
    :param matching_df:
    :param non_matching_df:
    :return:
    """

    for idx, row in img_pairs_by_indices.iterrows():
        if len(row) == 3:
            person_name = row[0]
            first_image_idx = row[1]
            second_image_idx = row[2]

            im1 = get_single_image(person_name, first_image_idx)
            im2 = get_single_image(person_name, second_image_idx)
        else:
            first_person_name = row[0]
            first_person_image_idx = row[1]
            second_person_name = row[2]
            second_person_image_idx = row[3]

            im1 = get_single_image(first_person_name, first_person_image_idx)
            im2 = get_single_image(second_person_name, second_person_image_idx)

        yield tf.convert_to_tensor(im1), tf.convert_to_tensor(im2)


    # for idx, row in img_pairs_by_indices.iterrows():
    #     if len(row) == 3:
    #         person_name = row[0]
    #         first_image_idx = row[1]
    #         second_image_idx = row[2]
    #
    #         im1 = get_single_image(person_name, first_image_idx)
    #         im2 = get_single_image(person_name, second_image_idx)
    #     else:
    #         first_person_name = row[0]
    #         first_person_image_idx = row[1]
    #         second_person_name = row[2]
    #         second_person_image_idx = row[3]
    #
    #         im1 = get_single_image(first_person_name, first_person_image_idx)
    #         im2 = get_single_image(second_person_name, second_person_image_idx)
    #
    #     return tf.convert_to_tensor(im1), tf.convert_to_tensor(im2)

def make_dataset(batch_size, seed, validation_split=0.2):
    def parse_single_image(image_path: str,
                           im_format: str = 'jpg') -> np.array:
        """
        Reads a single image file from the given images root directory.
        the function assumes each image is within a directory by the name of the person.
        the function assumes each image filename is <name>_####.<im_format>.
        the funciton supports the following formats ['jpg', 'png', 'jpeg'] and
        builds upon CV2.imread() method.
        if needed, the function returns the flatten image np.array (default is not to flatten)

        :param image_path: str, name of the person as it appears in the dataset
        :param im_format: str, format of the image file
        :return: np.array, the image.
        """
        assert im_format in ['jpg', 'png', 'jpeg']

        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=1)

        # convert unit8 tensor to floats in the [0,1] range
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.cast(image / 255., tf.float32)

        return image

    def load_images_paris_as_tensors(img_pairs):
        """
        :param img_pairs:
        :return:
        """
        return parse_single_image(img_pairs[0]), parse_single_image(img_pairs[1])

    def generate_image_name_path(image_root_path, name, idx):
        return os.sep.join([image_root_path, name, f'{name}_{str(idx).zfill(4)}.jpg'])

    def generate_matching_images_pairs_paths(image_pairs, image_root_path='lfw2Data/lfw2'):
        return [(generate_image_name_path(image_root_path, row[0], row[1]), generate_image_name_path(image_root_path, row[0], row[2])) for row in list(image_pairs.itertuples(index=False, name=None))]

    def generate_non_matching_images_pairs_paths(image_pairs, image_root_path='lfw2Data/lfw2'):
        return [(generate_image_name_path(image_root_path, row[0], row[1]),
                 generate_image_name_path(image_root_path, row[2], row[3])) for row in
                list(image_pairs.itertuples(index=False, name=None))]

    def configure_for_performance(ds, is_training=False):
        if is_training:
            ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(batch_size)
        ds = ds.cache()
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    # create train-validation datasets
    train_pairs_file_path = "/lfw2Data/pairsDevTrain.txt"

    train_matching_pairs, train_non_matching_pairs, number_of_pairs = get_matching_non_matching_pairs(file_path(train_pairs_file_path))

    X_train_matching_pairs, X_val_matching_pairs, y_train_matching_pairs, y_val_matching_pairs = train_test_split(train_matching_pairs, np.ones(len(train_matching_pairs)), test_size=validation_split, shuffle=True, random_state=seed)
    X_train_non_matching_pairs, X_val_non_matching_pairs, y_train_non_matching_pairs, y_val_non_matching_pairs = train_test_split(train_non_matching_pairs, np.zeros(len(train_non_matching_pairs)), test_size=validation_split, shuffle=True, random_state=seed)

    y_train = np.concatenate([y_train_matching_pairs, y_train_non_matching_pairs])
    y_val = np.concatenate([y_val_matching_pairs, y_val_non_matching_pairs])

    train_matching_images_pairs_paths = generate_matching_images_pairs_paths(X_train_matching_pairs)
    train_non_matching_images_pairs_paths = generate_non_matching_images_pairs_paths(X_train_non_matching_pairs)

    train_matching_pairs_ds = tf.data.Dataset.from_tensor_slices(train_matching_images_pairs_paths + train_non_matching_images_pairs_paths)
    train_ds = train_matching_pairs_ds.map(load_images_paris_as_tensors,
                                           num_parallel_calls=tf.data.experimental.AUTOTUNE)

    y_train = tf.data.Dataset.from_tensor_slices(y_train)
    train_ds = tf.data.Dataset.zip((train_ds, y_train))
    train_ds = configure_for_performance(train_ds, is_training=True)

    val_matching_images_pairs_paths = generate_matching_images_pairs_paths(X_val_matching_pairs)
    val_non_matching_images_pairs_paths = generate_non_matching_images_pairs_paths(X_val_non_matching_pairs)

    val_matching_pairs_ds = tf.data.Dataset.from_tensor_slices(val_matching_images_pairs_paths + val_non_matching_images_pairs_paths)
    val_ds = val_matching_pairs_ds.map(load_images_paris_as_tensors,
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    y_val = tf.data.Dataset.from_tensor_slices(y_val)
    val_ds = tf.data.Dataset.zip((val_ds, y_val))
    val_ds = configure_for_performance(val_ds)

    # create test dataset
    test_pairs_file_path = "/lfw2Data/pairsDevTest.txt"
    test_matching_pairs, test_non_matching_pairs, number_of_pairs = get_matching_non_matching_pairs(file_path(test_pairs_file_path))

    y_test = np.concatenate([np.ones(number_of_pairs), np.zeros(number_of_pairs)])

    test_matching_images_pairs_paths = generate_matching_images_pairs_paths(test_matching_pairs)
    test_non_matching_images_pairs_paths = generate_non_matching_images_pairs_paths(test_non_matching_pairs)

    test_matching_pairs_ds = tf.data.Dataset.from_tensor_slices(test_matching_images_pairs_paths + test_non_matching_images_pairs_paths)
    test_ds = test_matching_pairs_ds.map(load_images_paris_as_tensors,
                                         num_parallel_calls=tf.data.experimental.AUTOTUNE)

    y_test = tf.data.Dataset.from_tensor_slices(y_test)
    test_ds = tf.data.Dataset.zip((test_ds, y_test))
    test_ds = configure_for_performance(test_ds)

    return train_ds, val_ds, test_ds
