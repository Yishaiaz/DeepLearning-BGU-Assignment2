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

    def test_load_images_as_vectors(self):
        matching_df = pd.DataFrame({"name": ['Aaron_Peirsol'], 'n1': [1], 'n2': [2]})
        non_matching_df = pd.DataFrame()

        org_image1 = cv2.imread('lfw2Data/lfw2/Aaron_Peirsol/Aaron_Peirsol_0001.jpg')
        org_image2 = cv2.imread('lfw2Data/lfw2/Aaron_Peirsol/Aaron_Peirsol_0002.jpg')

        image_pairs, pairs_labels = load_images_as_vectors(matching_df, non_matching_df)
        im1_p1 = image_pairs[0][0]
        im2_p1 = image_pairs[0][1]
        label = pairs_labels[0]
        self.assertEqual(1, label)
        np.testing.assert_array_almost_equal(im1_p1, org_image1)
        np.testing.assert_array_almost_equal(im2_p1, org_image2)

        matching_df = pd.DataFrame()
        non_matching_df = pd.DataFrame({"name1": ['Aaron_Peirsol'], 'n1': [3], 'name2': ['Abdoulaye_Wade'], 'n2': [1]})

        org_image1 = cv2.imread('lfw2Data/lfw2/Aaron_Peirsol/Aaron_Peirsol_0003.jpg')
        org_image2 = cv2.imread('lfw2Data/lfw2/Abdoulaye_Wade/Abdoulaye_Wade_0001.jpg')

        image_pairs, pairs_labels = load_images_as_vectors(matching_df, non_matching_df)
        im1_p1 = image_pairs[0][0]
        im1_p2 = image_pairs[0][1]
        label = pairs_labels[0]
        self.assertEqual(0, label)
        np.testing.assert_array_almost_equal(im1_p1, org_image1)
        np.testing.assert_array_almost_equal(im1_p2, org_image2)

        pairs_file_path = 'lfw2Data/pairsDevTest.txt'
        matching_df, non_matching_df, number_of_pairs = get_matching_non_matching_pairs(pairs_file_path)
        image_pairs, pairs_labels = load_images_as_vectors(matching_df, non_matching_df)
        self.assertEqual(image_pairs.shape[0], pairs_labels.shape[0])
