import os
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


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

    matching_pairs = {'name1': [], 'n1': [], 'name2': [], 'n2': []}
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
                matching_pairs['name1'].append(split_single_line[0])
                matching_pairs['n1'].append(int(split_single_line[1]))
                matching_pairs['name2'].append(split_single_line[0])
                matching_pairs['n2'].append(int(split_single_line[2]))
            # non-matching pairs - only 4 entries in line
            else:
                non_matching_pairs['name1'].append(split_single_line[0])
                non_matching_pairs['n1'].append(int(split_single_line[1]))
                non_matching_pairs['name2'].append(split_single_line[2])
                non_matching_pairs['n2'].append(int(split_single_line[3]))

            single_line = pairs_file.readline()

    return pd.DataFrame(matching_pairs), pd.DataFrame(non_matching_pairs), number_of_pairs


def make_dataset(images_directory: str,
                 resize_dim: Tuple[int, int],
                 batch_size: int = 32,
                 val_size: float = 0.2,
                 augment_training_dataset: bool = False,
                 use_transfer_learning_architecture: bool = False,
                 n: int = 3,
                 train_pairs_file_path: str = 'lfw2Data/pairsDevTrain.txt',
                 test_pairs_file_path: str = 'lfw2Data/pairsDevTest.txt',
                 seed: int = 0):
    """
    This function generates training, validation and test datasets.
    Also, generates one-shot validation/test accuracy tests configured by the given n parameter which defines how many n-way to create.
    """

    def configure_for_performance(ds, batchsize: int, is_training=False):
        ds = ds.cache()
        if is_training:
            ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(batchsize)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    def read_image(image_path: str):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3 if use_transfer_learning_architecture else 1)
        if resize_dim[0] != 250:
            image = tf.image.resize(image, [resize_dim[0], resize_dim[1]])
        image = tf.cast(image / 255, tf.float32)
        image = tf.image.convert_image_dtype(image, tf.float32)

        return image

    def load_images_pairs_as_tensors(img_pairs):
        return read_image(img_pairs[0]), read_image(img_pairs[1])

    def get_full_image_path(images_root_path, name, idx):
        image_name = '{name}_{id}.jpg'.format(name=name, id=str(idx).zfill(4))
        image_full_path = os.sep.join([images_root_path, name, image_name])
        return image_full_path

    def get_all_image_paths_in_df(df, main_dir) -> np.array:
        pairs_image_paths = {'name1': [], 'n1': [], 'name2': [], 'n2': []}

        for idx, row in df.iterrows():
            person1_name = row[0]
            first_image_idx = row[1]
            person2_name = row[2]
            second_image_idx = row[3]

            im1_path = get_full_image_path(main_dir, person1_name, first_image_idx)
            im2_path = get_full_image_path(main_dir, person2_name, second_image_idx)

            pairs_image_paths['name1'].append(person1_name)
            pairs_image_paths['n1'].append(im1_path)
            pairs_image_paths['name2'].append(person2_name)
            pairs_image_paths['n2'].append(im2_path)

        return pd.DataFrame(pairs_image_paths)

    def create_labels_vector(vector: np.array, label: int):
        return np.zeros(len(vector)) + label

    def get_file_x_y(pairs_file_path):
        matching_df, non_matching_df, number_of_pairs = get_matching_non_matching_pairs(pairs_file_path)
        matching_pairs_image_paths = get_all_image_paths_in_df(df=matching_df, main_dir=images_directory)
        matching_images_labels = create_labels_vector(matching_pairs_image_paths, label=1).reshape(-1, 1)
        non_matching_pairs_image_paths = get_all_image_paths_in_df(df=non_matching_df, main_dir=images_directory)
        non_matching_images_labels = create_labels_vector(non_matching_pairs_image_paths, label=0).reshape(-1, 1)

        return matching_pairs_image_paths, matching_images_labels, non_matching_pairs_image_paths, non_matching_images_labels

    def generate_one_shot_dataset(X_matching, X_non_matching):
        one_shot_tests = []

        for idx, row in X_matching.iterrows():
            one_shot_test = []
            person_name = row[0]
            first_image_idx = row[1]
            second_image_idx = row[3]

            one_shot_test.append((first_image_idx, second_image_idx))

            pairs_including_current_person_from_non_matching_df = X_non_matching.loc[(X_non_matching['name1'] == person_name) & (X_non_matching['name2'] == person_name)]

            for idx_non_matching, row_non_matching in pairs_including_current_person_from_non_matching_df.iterrows():
                one_shot_test.append((row_non_matching[1], row_non_matching[3]))

            if len(one_shot_test) <= n:
                pairs_not_including_current_person_from_non_matching_df = X_non_matching.loc[(X_non_matching['name1'] != person_name) & (X_non_matching['name2'] != person_name)]
                non_matching_persons = set()

                while len(one_shot_test) < n:
                    sample_pair_not_including_current_person_from_non_matching_df = pairs_not_including_current_person_from_non_matching_df.sample(1, random_state=seed)
                    candidate1 = sample_pair_not_including_current_person_from_non_matching_df.iloc[0]['n1']
                    candidate2 = sample_pair_not_including_current_person_from_non_matching_df.iloc[0]['n2']
                    if candidate1 not in non_matching_persons:
                        one_shot_test.append((first_image_idx, candidate1))
                        non_matching_persons.add(candidate1)

                    if len(one_shot_test) < n and candidate2 not in non_matching_persons:
                        one_shot_test.append((first_image_idx, candidate2))
                        non_matching_persons.add(candidate2)

            one_shot_tests.append(np.array(one_shot_test))

        return np.array(one_shot_tests)

    def get_file_train_test_split(matching_pairs_image_paths, matching_images_labels, non_matching_pairs_image_paths, non_matching_images_labels):
        # matching
        X_matching_train, X_matching_val, y_matching_train, y_matching_val = train_test_split(matching_pairs_image_paths, matching_images_labels,
                                                                                              test_size=val_size,
                                                                                              shuffle=True,
                                                                                              random_state=seed)

        # non-matching
        X_non_matching_train, X_non_matching_val, y_non_matching_train, y_non_matching_val = train_test_split(non_matching_pairs_image_paths, non_matching_images_labels,
                                                                                                              test_size=val_size,
                                                                                                              shuffle=True,
                                                                                                              random_state=seed)

        training_image_paths = np.vstack((X_matching_train[['n1', 'n2']].to_numpy(), X_non_matching_train[['n1', 'n2']].to_numpy()))
        training_labels = np.vstack((y_matching_train, y_non_matching_train))

        # create verification validation set
        val_image_paths = np.vstack((X_matching_val[['n1', 'n2']].to_numpy(), X_non_matching_val[['n1', 'n2']].to_numpy()))
        val_labels = np.vstack((y_matching_val, y_non_matching_val))

        # create one-shot validation set
        validation_one_shot_tests = generate_one_shot_dataset(X_matching_val, X_non_matching_val)

        return training_image_paths, training_labels, val_image_paths, val_labels, validation_one_shot_tests

    def generate_augmented_dataset(image_paths, augmentation_name, ds, labels_ds):
        image_augmented_paths = []

        for image_path in image_paths:
            image_path1_splitted = image_path[0].split(os.path.sep)
            image_path2_splitted = image_path[1].split(os.path.sep)
            image_path1_augmented = os.sep.join(
                ["lfw2Data/augmentations", augmentation_name, image_path1_splitted[-1]])
            image_path2_augmented = os.sep.join(
                ["lfw2Data/augmentations", augmentation_name, image_path2_splitted[-1]])
            image_augmented_paths.append((image_path1_augmented, image_path2_augmented))

        augmented_ds = tf.data.Dataset.from_tensor_slices(np.array(image_augmented_paths))
        augmented_ds = augmented_ds.map(load_images_pairs_as_tensors, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return ds.concatenate(augmented_ds), labels_ds.concatenate(labels_ds)

    def turn_to_zipped_ds(image_paths, labels, is_training=False):
        ds = tf.data.Dataset.from_tensor_slices(image_paths)
        ds = ds.map(load_images_pairs_as_tensors, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        labels_ds = tf.data.Dataset.from_tensor_slices(labels)

        if augment_training_dataset and is_training:
            ds, labels_ds = generate_augmented_dataset(image_paths, "noise", ds, labels_ds)
            ds, labels_ds = generate_augmented_dataset(image_paths, "rotation45", ds, labels_ds)
            ds, labels_ds = generate_augmented_dataset(image_paths, "center_crop", ds, labels_ds)
            ds, labels_ds = generate_augmented_dataset(image_paths, "flip_left_right", ds, labels_ds)
            ds, labels_ds = generate_augmented_dataset(image_paths, "noise_and_center_crop", ds, labels_ds)

            ds = tf.data.Dataset.zip((ds, labels_ds))
        else:
            ds = tf.data.Dataset.zip((ds, labels_ds))
        ds = configure_for_performance(ds, batchsize=batch_size, is_training=is_training)
        return ds

    def turn_to_zipped_one_shot_ds(one_shot_tests):
        # create dataset per test configuration
        one_shot_tests_ds = []
        for one_shot_test in one_shot_tests:
            ds = tf.data.Dataset.from_tensor_slices(one_shot_test)
            ds = ds.map(load_images_pairs_as_tensors, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            ds = configure_for_performance(ds, batchsize=batch_size, is_training=False)

            labels = np.zeros(len(one_shot_test)).reshape(-1, 1)
            labels[0][0] = 1

            ds = tf.data.Dataset.zip((ds, tf.data.Dataset.from_tensor_slices(labels)))
            one_shot_tests_ds.append(ds)

        return one_shot_tests_ds

    # train set
    matching_pairs_image_paths, matching_images_labels, non_matching_pairs_image_paths, non_matching_images_labels = get_file_x_y(train_pairs_file_path)

    training_image_paths, training_labels, val_image_paths, val_labels, validation_one_shot_tests = \
        get_file_train_test_split(matching_pairs_image_paths,
                                  matching_images_labels,
                                  non_matching_pairs_image_paths,
                                  non_matching_images_labels)

    training_ds = turn_to_zipped_ds(training_image_paths, training_labels, is_training=True)

    val_ds = turn_to_zipped_ds(val_image_paths, val_labels)

    one_shot_val_ds_list = turn_to_zipped_one_shot_ds(validation_one_shot_tests)

    # test set
    matching_pairs_image_paths, matching_images_labels, non_matching_pairs_image_paths, non_matching_images_labels = \
        get_file_x_y(test_pairs_file_path)
    # we don't mind not shuffling as this is the test
    test_image_paths = np.vstack((matching_pairs_image_paths[['n1', 'n2']].to_numpy(), non_matching_pairs_image_paths[['n1', 'n2']].to_numpy()))
    test_labels = np.vstack((matching_images_labels, non_matching_images_labels))

    test_ds = turn_to_zipped_ds(test_image_paths, test_labels)

    # create one-shot test set
    test_one_shot_tests = generate_one_shot_dataset(matching_pairs_image_paths, non_matching_pairs_image_paths)
    one_shot_test_ds_list = turn_to_zipped_one_shot_ds(test_one_shot_tests)

    return training_ds, val_ds, test_ds, one_shot_val_ds_list, one_shot_test_ds_list
