
def generate_n_way_oneshot_accuracy_test(dataset: tf.data.Dataset, trained_sets: dict, N: int = 5) -> Tuple[np.array, np.array]:

    def collect_all_pairs_from_paths_ds(full_dataset):
        matching_image_pairs, non_matching_image_pairs = [], []
        for batch in full_dataset:
            paths_in_batch = batch[0]
            labels_in_batch = batch[1]
            for i in range(len(labels_in_batch)):
                # converting the bytes array to plain str for ease of use
                if labels_in_batch[i] == 1:
                    matching_image_pairs.append((tensor_bytes_to_str(paths_in_batch[0][i]), tensor_bytes_to_str(paths_in_batch[1][i])))
                else:
                    non_matching_image_pairs.append((tensor_bytes_to_str(paths_in_batch[0][i]), tensor_bytes_to_str(paths_in_batch[1][i])))

        return matching_image_pairs, non_matching_image_pairs

    def tensor_bytes_to_str(tensor_bytes: tf.Tensor):
        return tf.compat.as_str_any(tensor_bytes.numpy())

    def choose_random_from_matching(matching_paths):
        rand_idx = int(np.random.rand(1)[0] * len(matching_paths))
        path_to_im1, path_to_im2 = matching_paths[rand_idx][0], matching_paths[rand_idx][1]
        # name = path_to_im1.split(os.sep)[-2]
        return path_to_im1, path_to_im2

    def check_if_pair_was_trained(pair_of_paths):
        def check_if_trained_with_image(image_path: str, images_trained_lst: list):
            return image_path in images_trained_lst
        trained_with_image1 = trained_sets.get(pair_of_paths[0], [])
        trained_with_image2 = trained_sets.get(pair_of_paths[1], [])
        return check_if_trained_with_image(pair_of_paths[0], trained_with_image1) and \
               check_if_trained_with_image(pair_of_paths[1], trained_with_image2)

    def gather_non_matching_from_ds(target_image_path, non_matching_pairs):
        non_matching_pairs_to_test = []
        for image_path1, image_path2 in non_matching_pairs:
            if len(non_matching_pairs_to_test) >= N-1: #todo remove this condition
                break
            if not check_if_pair_was_trained((image_path1, target_image_path)) and target_image_path != image_path1:
                non_matching_pairs_to_test.append((target_image_path, image_path1))
            if len(non_matching_pairs_to_test) >= N-1:
                break
            if not check_if_pair_was_trained((image_path2, target_image_path)) and target_image_path != image_path2:
                non_matching_pairs_to_test.append((target_image_path, image_path2))

        return non_matching_pairs_to_test

    input_pairs, labels = [], []

    # read the dataset - assuming this is the validation/test set only
    matching_image_paths_pairs, non_matching_image_paths_pairs = collect_all_pairs_from_paths_ds(dataset)

    # choose a random image path from the matching pairs
    path_to_im1, path_to_im2 = choose_random_from_matching(matching_image_paths_pairs)

    # add the matching tuple to the test set
    input_pairs.append((path_to_im1, path_to_im2))
    labels.append(1)

    # choose one of the images fo the chosen pair and gather all non matching images
    rand_idx = int(np.random.rand(1)[0] * 2)
    # todo derive random matches from entire dataset
    non_matching_to_test = gather_non_matching_from_ds([path_to_im1, path_to_im2][rand_idx], non_matching_image_paths_pairs)
    input_pairs += non_matching_to_test
    labels += np.zeros(len(non_matching_to_test), dtype=int).tolist()
    # todo: fill up to N if len(labels) < N - no need
    # convert all to tf tensor datasets
    input_pairs, labels = tf.data.Dataset.from_tensor_slices(np.array(input_pairs)), \
                          tf.data.Dataset.from_tensor_slices(np.array(labels))

    return input_pairs, labels # todo: isnt it better if the func returns the dataset as a single object (X, Y)?

