import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
import skimage.transform
import skimage.util
import tensorflow as tf


pairs_file_path = 'lfw2Data/pairsDevTrain.txt'
images_aumentation_directory = 'lfw2Data/augmentations'
images_directory = 'lfw2Data/lfw2'


def noise(images):
    return skimage.util.random_noise(images[0].numpy(), seed=0, mode='s&p'), skimage.util.random_noise(images[1].numpy(), seed=0, mode='s&p')


def rotation45(images):
    return tf.convert_to_tensor(skimage.transform.rotate(images[0].numpy(), [45, -45][random.randint(0, 1)])), tf.convert_to_tensor(skimage.transform.rotate(images[1].numpy(), [45, -45][random.randint(0, 1)]))


def center_crop(images):
    return tf.image.resize_with_crop_or_pad(tf.image.central_crop(images[0], 0.5), 250, 250), tf.image.resize_with_crop_or_pad(tf.image.central_crop(images[1], 0.5), 250, 250)


def flip_left_right(images):
    return tf.image.flip_left_right(images[0]), tf.image.flip_left_right(images[1])


def noise_and_center_crop(images):
    return center_crop(noise(images))


def read_image(image_path: str):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.cast(image / 255, tf.float32)
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image


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


def apply_transformation_and_save(images_path, augmentation_name, image1, image2, augmentation_func):
    image_full_path1 = os.sep.join([images_aumentation_directory, augmentation_name, images_path[0].split(os.path.sep)[-1]])
    image_full_path2 = os.sep.join([images_aumentation_directory, augmentation_name, images_path[1].split(os.path.sep)[-1]])

    image1_transformed, image2_transformed = augmentation_func((image1, image2))

    tf.keras.preprocessing.image.save_img(image_full_path1, image1_transformed)
    tf.keras.preprocessing.image.save_img(image_full_path2, image2_transformed)


def generate_augmentation():
    matching_df, non_matching_df, number_of_pairs = get_matching_non_matching_pairs(pairs_file_path)
    matching_image_paths = get_all_image_paths_in_df(df=matching_df, main_dir=images_directory)
    non_matching_image_paths = get_all_image_paths_in_df(df=non_matching_df, main_dir=images_directory)

    training_image_paths = np.vstack((matching_image_paths, non_matching_image_paths))

    for images_path in training_image_paths:
        image1 = read_image(images_path[0])
        image2 = read_image(images_path[1])

        # noise
        apply_transformation_and_save(images_path, "noise", image1, image2, noise)

        # rotation45
        apply_transformation_and_save(images_path, "rotation45", image1, image2, rotation45)

        # center_crop
        apply_transformation_and_save(images_path, "center_crop", image1, image2, center_crop)

        # flip_left_right
        apply_transformation_and_save(images_path, "flip_left_right", image1, image2, flip_left_right)

        # noise_and_center_crop
        apply_transformation_and_save(images_path, "noise_and_center_crop", image1, image2, noise_and_center_crop)


if __name__ == "__main__":
    generate_augmentation()

# TODO remove before submission
# image_path_1 = file_path("lfw2Data/lfw2/Aaron_Eckhart/Aaron_Eckhart_0001.jpg")
# image_1 = read_image(image_path_1)
#
# image_path_2 = file_path("lfw2Data/lfw2/Abba_Eban/Abba_Eban_0001.jpg")
# image_2 = read_image(image_path_2)
#
# seed = 45
# random.seed(seed)
#
# images = (image_1, image_2)
#
# images1 = tf.image.flip_left_right(images)
# plt.subplot(2,1,1)
# plt.imshow(images1[0])
# plt.subplot(2,1,2)
# plt.imshow(images1[1])
# plt.show()
#
# images2 = center_crop(images)
# plt.subplot(2,1,1)
# plt.imshow(images2[0])
# plt.subplot(2,1,2)
# plt.imshow(images2[1])
# plt.show()
#
# images3 = noise_and_center_crop(images, seed)
# plt.subplot(2,1,1)
# plt.imshow(images3[0])
# plt.subplot(2,1,2)
# plt.imshow(images3[1])
# plt.show()
#
# images4 = noise(images, seed)
# plt.subplot(2,1,1)
# plt.imshow(images4[0])
# plt.subplot(2,1,2)
# plt.imshow(images4[1])
# plt.show()
#
# images5 = rotation45(images)
# plt.subplot(2,1,1)
# plt.imshow(images5[0])
# plt.subplot(2,1,2)
# plt.imshow(images5[1])
# plt.show()