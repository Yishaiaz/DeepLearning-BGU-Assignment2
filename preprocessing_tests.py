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

