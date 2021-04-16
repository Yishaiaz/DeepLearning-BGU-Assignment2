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
        while single_line != '' and single_line != None:
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
                 batch_size: int = 32,
                 val_size: float = 0.2,
                 augment_training_dataset: bool = False,
                 seed: int = 42):

    def configure_for_performance(ds, batchsize: int, is_training=False):
        if is_training:
            ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(batchsize)
        ds = ds.cache()
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    def read_image(image_path: str):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=1)
        image = tf.cast(image / 255, tf.float32)
        image = tf.image.convert_image_dtype(image, tf.float32)

        return image

    def load_images_pairs_as_tensors(img_pairs):
        """
        :param img_pairs:
        :return:
        """
        return read_image(img_pairs[0]), read_image(img_pairs[1])

    def get_full_image_path(images_root_path, name, idx):
        image_name = '{name}_{id}.jpg'.format(name=name, id=str(idx).zfill(4))
        image_full_path = os.sep.join([images_root_path, name, image_name])
        return image_full_path

    def get_all_image_paths_in_df(df, main_dir) -> np.array:
        images_paths = []
        for idx, row in df.iterrows():
            person1_name = row[0]
            first_image_idx = row[1]
            person2_name = row[2]
            second_image_idx = row[3]

            im1_path = get_full_image_path(main_dir, person1_name, first_image_idx)
            im2_path = get_full_image_path(main_dir, person2_name, second_image_idx)

            images_paths.append((im1_path, im2_path))

        return np.array(images_paths)

    def create_labels_vector(vector:np.array, label: int):
        return np.zeros(len(vector)) + label

    def get_file_x_y(pairs_file_path):
        matching_df, non_matching_df, number_of_pairs = get_matching_non_matching_pairs(pairs_file_path)
        matching_image_paths = get_all_image_paths_in_df(df=matching_df, main_dir=images_directory)
        matching_images_labels = create_labels_vector(matching_image_paths, label=1).reshape(-1, 1)
        non_matching_image_paths = get_all_image_paths_in_df(df=non_matching_df, main_dir=images_directory)
        non_matching_images_labels = create_labels_vector(non_matching_image_paths, label=0).reshape(-1, 1)

        return matching_image_paths, matching_images_labels, non_matching_image_paths, non_matching_images_labels

    def get_file_train_test_split(matching_image_paths, matching_images_labels, non_matching_image_paths, non_matching_images_labels):
        # matching
        X_matching_train, X_matching_val, y_matching_train, y_matching_val = train_test_split(matching_image_paths, matching_images_labels,
                                                                                              test_size=val_size,
                                                                                              shuffle=True,
                                                                                              random_state=seed)

        # non-matching
        X_non_matching_train, X_non_matching_val, y_non_matching_train, y_non_matching_val = train_test_split(non_matching_image_paths, non_matching_images_labels,
                                                                                                              test_size=val_size,
                                                                                                              shuffle=True,
                                                                                                              random_state=seed)

        training_image_paths = np.vstack((X_matching_train, X_non_matching_train))
        training_labels = np.vstack((y_matching_train, y_non_matching_train))

        val_image_paths = np.vstack((X_matching_val, X_non_matching_val))
        val_labels = np.vstack((y_matching_val, y_non_matching_val))
        return training_image_paths, training_labels, val_image_paths, val_labels

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
            ds = tf.data.Dataset.zip((ds, tf.data.Dataset.from_tensor_slices(labels)))
        ds = configure_for_performance(ds, batchsize=batch_size, is_training=is_training)
        return ds

    # train set
    pairs_file_path = 'lfw2Data/pairsDevTrain.txt'
    matching_image_paths, matching_images_labels, non_matching_image_paths, non_matching_images_labels = \
        get_file_x_y(pairs_file_path)

    training_image_paths, training_labels, val_image_paths, val_labels = \
        get_file_train_test_split(matching_image_paths,
                                  matching_images_labels,
                                  non_matching_image_paths,
                                  non_matching_images_labels)

    training_ds = turn_to_zipped_ds(training_image_paths, training_labels, is_training=True)

    val_ds = turn_to_zipped_ds(val_image_paths, val_labels)

    # test set
    pairs_file_path = 'lfw2Data/pairsDevTest.txt'
    matching_image_paths, matching_images_labels, non_matching_image_paths, non_matching_images_labels = \
        get_file_x_y(pairs_file_path)
    # we don't mind not shuffling as this is the test
    test_image_paths = np.vstack((matching_image_paths, non_matching_image_paths))
    test_labels = np.vstack((matching_images_labels, non_matching_images_labels))

    test_ds = turn_to_zipped_ds(test_image_paths, test_labels)

    return training_ds, val_ds, test_ds

def generate_n_way_oneshot_accuracy_test(dataset: tf.data.Dataset,
                                         name_to_idxs_in_val: dict,
                                         trained_sets: dict,
                                         N: int = 5) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    generates two tf.data.Dataset objects containing the image pairs and labels.
    all of first images within the pairs are of the same person, and all of the 2nd
    images (excluding the first pair) are of different people.
    Accordingly, all labels are set to 0 (non-matching) excluding the first pair in the set,
    set to 1 (matching).
    Example for name_to_idxs_in_val dict:
        name_to_idxs_in_val = {
        (SampleIDX, colIDX)
        'Abdoulaye_Wade': [(0, 0), (0, 1), (1, 0)],
        'Adam_Ant': [(2, 0)],
        'Alvaro_Uribe': [(3, 0), (3, 1)],
        'Abdulaziz_Kamilov': [(1, 1)],
        'Adam_Scott': [(2, 0)]
    }
    Example for trained_sets dict:
    trained_sets = {
        'Abdoulaye_Wade': ['Adam_Ant', 'Alvaro_Uribe', 'Adam_Scott']
    }

    :param dataset: a tf.data.BatchedDataset - the validation/test set for a simple accuracy metric.
    :param name_to_idxs_in_val: dictionary - holds the images indices in the validation/test dataset
    (all batches are flattened).
    :param trained_sets: dictioinary - holds the already trained pairs by person name.
    :param N: int - the size of the n-way testset
    :return: Tuple[tf.data.Dataset - image pairs, tf.data.Dataset - labels]
    """
    def get_person_name_by_indices(sample_idx:int, col_idx:int, names_to_indices: dict):
        indices_tuple = (sample_idx, col_idx)
        # iterate over names:
        for name, indices_lst in names_to_indices.items():
            if indices_tuple in indices_lst:
                return name

    def choose_random_from_matching(all_images_ds: tf.data.Dataset):
        sample_idx = 0
        random_match = None
        for batch in all_images_ds:
            images_in_batch = batch[0]
            labels_in_batch = batch[1]
            for i in range(len(labels_in_batch)):
                if labels_in_batch[i] == 1:
                    if np.random.rand(1) >= 0.5:
                        im1 = images_in_batch[i][0]
                        im2 = images_in_batch[i][1]
                        return im1, im2, sample_idx
                sample_idx += 1
        # if all randoms < 0.5, pick the first matching
        sample_idx = 0
        if random_match is None:
            for batch in all_images_ds:
                images_in_batch = batch[0]
                labels_in_batch = batch[1]
                for i in range(len(labels_in_batch)):
                    if labels_in_batch[i] == 1:
                        im1 = images_in_batch[i][0]
                        im2 = images_in_batch[i][1]
                        return im1, im2, sample_idx
                    sample_idx += 1

        return random_match

    def check_if_pair_was_trained(person_name1:str, person_name2:str):
        def check_if_trained_with_image(name: str, images_trained_lst: list):
            return name in images_trained_lst
        trained_with_image1 = trained_sets.get(person_name1, [])
        trained_with_image2 = trained_sets.get(person_name2, [])
        return check_if_trained_with_image(person_name2, trained_with_image1) or \
               check_if_trained_with_image(person_name1, trained_with_image2)

    def get_image_by_idx(indices: Tuple[int, int]) -> tf.Tensor:
        target_sample_idx, target_col_idx = indices[0], indices[1]
        sample_idx = 0
        for batch in dataset:
            images_in_batch = batch[0]
            labels_in_batch = batch[1]
            for i in range(len(labels_in_batch)):
                if target_sample_idx == sample_idx:
                    return images_in_batch[i][target_col_idx]
                sample_idx += 1

    def gather_non_matching_from_ds(image_to_compare: tf.Tensor,
                                    all_dataset: tf.data.Dataset,
                                    person_name: str,
                                    name_to_indices: dict):


        # remove the indices of the person from the possible choices:
        name_to_indices.pop(person_name)
        non_matching_images = []

        while len(non_matching_images) < N-1:
            # take a random name of the remaining names.
            random_key = list(name_to_indices.keys())[int(np.random.rand(1) * len(name_to_indices.keys()))]
            # check if both were already seen in training (prevent leakage)
            if not check_if_pair_was_trained(person_name1=person_name, person_name2=random_key):
                # get an image of the 2nd person (random key)
                random_im_indices = name_to_indices[random_key][int(np.random.rand(1)*len(name_to_indices[random_key]))]
                image2_to_compare = get_image_by_idx(random_im_indices)

                non_matching_images.append((image_to_compare, image2_to_compare))

        return non_matching_images

    input_pairs, labels = [], []

    # read the dataset - assuming this is the validation/test set only
    # choose a random image path from the matching pairs
    matching_im1, matching_im2, sample_idx = choose_random_from_matching(dataset)

    # add the matching tuple to the test set
    input_pairs.append((matching_im1, matching_im2))
    labels.append(1)

    # choose one of the images of the chosen pair and gather all non matching images
    rand_col_idx = int(np.random.rand(1)[0] * 2)
    name_of_person = get_person_name_by_indices(sample_idx=sample_idx,
                                                col_idx=rand_col_idx,
                                                names_to_indices=name_to_idxs_in_val)

    non_matching_to_test = gather_non_matching_from_ds(image_to_compare=[matching_im1, matching_im2][rand_col_idx],
                                                       all_dataset=dataset,
                                                       person_name=name_of_person,
                                                       name_to_indices=name_to_idxs_in_val)
    input_pairs += non_matching_to_test
    labels += np.zeros(len(non_matching_to_test), dtype=int).tolist()
    # convert all to tf tensor datasets
    input_pairs, labels = tf.data.Dataset.from_tensor_slices(np.array(input_pairs)), \
                          tf.data.Dataset.from_tensor_slices(np.array(labels))

    return input_pairs, labels

