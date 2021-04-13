


from unittest import TestCase
from preprocessing_utils import *
import cv2


class TestSiameseNetworkPreProcessing(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_get_matching_non_matching_pairs(self):
        # test the shape of the output - test
        pairs_file_path = 'lfw2Data/pairsDevTest.txt'
        matching_df, non_matching_df, number_of_pairs = get_matching_non_matching_pairs(pairs_file_path)

        self.assertEqual(matching_df.shape[1], 4)
        self.assertEqual(non_matching_df.shape[1], 4)
        self.assertEqual(matching_df.shape[0], number_of_pairs)
        self.assertEqual(non_matching_df.shape[0], number_of_pairs)

        # test the shape of the output - train
        pairs_file_path = 'lfw2Data/pairsDevTrain.txt'
        matching_df, non_matching_df, number_of_pairs = get_matching_non_matching_pairs(pairs_file_path)

        self.assertEqual(matching_df.shape[1], 4)
        self.assertEqual(non_matching_df.shape[1], 4)
        self.assertEqual(matching_df.shape[0], number_of_pairs)
        self.assertEqual(non_matching_df.shape[0], number_of_pairs)

    def test_generate_nway(self):
        images_paths = ['lfw2Data/lfw2/Abdoulaye_Wade/Abdoulaye_Wade_0001.jpg',
                        'lfw2Data/lfw2/Abdoulaye_Wade/Abdoulaye_Wade_0003.jpg',
                        'lfw2Data/lfw2/Adam_Ant/Adam_Ant_0001.jpg',
                        'lfw2Data/lfw2/Alvaro_Uribe/Alvaro_Uribe_0003.jpg'], ['lfw2Data/lfw2/Abdoulaye_Wade/Abdoulaye_Wade_0002.jpg',
                                                                              'lfw2Data/lfw2/Abdulaziz_Kamilov/Abdulaziz_Kamilov_0001.jpg',
                                                                              'lfw2Data/lfw2/Adam_Scott/Adam_Scott_0002.jpg',
                                                                              'lfw2Data/lfw2/Alvaro_Uribe/Alvaro_Uribe_0010.jpg']
        labels = [1, 0, 0, 1]
        image_paths_ds = tf.data.Dataset.from_tensor_slices((images_paths[0], images_paths[1]))
        labels_ds = tf.data.Dataset.from_tensor_slices(labels)
        test_ds = tf.data.Dataset.zip((image_paths_ds, labels_ds)).batch(2)
        # training_ds, val_ds, test_ds = make_dataset(images_directory='lfw2Data/lfw2')
        generate_n_way_oneshot_accuracy_test(test_ds, {})
        self.assertTrue(True)
