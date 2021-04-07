from unittest import TestCase
from network import *
import cv2


class TestSiameseNetwork(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_get_matching_non_matching_pairs(self):
        # test the shape of the output - test
        pairs_file_path = 'lfw2Data/pairsDevTest.txt'
        matching_df, non_matching_df, number_of_pairs = get_matching_non_matching_pairs(pairs_file_path)

        self.assertEqual(matching_df.shape[1], 3)
        self.assertEqual(non_matching_df.shape[1], 4)
        self.assertEqual(matching_df.shape[0], number_of_pairs)
        self.assertEqual(non_matching_df.shape[0], number_of_pairs)

        # test the shape of the output - train
        pairs_file_path = 'lfw2Data/pairsDevTrain.txt'
        matching_df, non_matching_df, number_of_pairs = get_matching_non_matching_pairs(pairs_file_path)

        self.assertEqual(matching_df.shape[1], 3)
        self.assertEqual(non_matching_df.shape[1], 4)
        self.assertEqual(matching_df.shape[0], number_of_pairs)
        self.assertEqual(non_matching_df.shape[0], number_of_pairs)

    def test_get_single_image(self):
        name, im_idx = 'Aaron_Eckhart', 1

        full_path_to_image = 'lfw2Data/lfw2/Aaron_Eckhart/Aaron_Eckhart_0001.jpg'
        true_im = cv2.imread(full_path_to_image)
        # test with no flattening
        np.testing.assert_array_almost_equal(get_single_image(name=name, image_idx=im_idx), true_im)
        # test with flattening
        np.testing.assert_array_almost_equal(get_single_image(name=name, image_idx=im_idx, flatten_im=True), true_im.flatten())
