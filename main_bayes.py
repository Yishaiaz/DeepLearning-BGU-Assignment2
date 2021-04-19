import contextlib
import csv
import numpy as np
import tensorflow as tf
from kerastuner import BayesianOptimization, HyperParameters
from tensorflow.python.keras.callbacks import EarlyStopping

from preprocessing_utils import make_dataset
from siamese_network import SiameseNeuralNetworkHyperModel
from utils import file_path, test_n_way, OneShotLearningAccuracyTestCallback


hp = HyperParameters()
hp.Choice('learning_rate', [5e-5, 1e-5, 5e-4, 1e-4, 1e-3, 5e-6])
hp.Choice('dense_layer_size', [1028, 4096, 512])
hp.Choice('enable_batch_normalization', [True, False])
hp.Choice('bias_initializer', ["default", "zeros"])
hp.Choice('conv2D_kernel_initializer', ["default", "he_normal"])
hp.Choice('dense_kernel_initializer', ["default", "he_normal"])
hp.Choice('dropout_rate', [0.0, 0.2, 0.5])
hp.Choice('distance_metric', ["abs", "euclidean_distance"])
hp.Choice('l2_regularizer', [-1.0, 0.01, 0.05, 0.1, 0.5])
hp.Choice('optimizer', ["adam", "sgd", "RMSprop"])


def main():
    global hp
    
    # configuration
    seed = 0
    mobilenet_input_shape = (224, 224, 3)
    images_directory = "./lfw2Data/lfw2/"
    project_name = "batch_32_without_data_augmentation"
    drive_prefix = "./"
    directory = drive_prefix + "/tuner_results"
    tf_log_dir = drive_prefix + "/tf_logs/tuner/" + project_name
    csv_file_name = directory + "/" + project_name + "/" + project_name + ".csv"

    augment_dataset = False
    overwrite = False
    run_bayes_search = False
    use_transfer_learning_architecture = True
    image_resize = False
    lfw_a_input_shape = (150, 150, 1) if image_resize else (250, 250, 1)
    input_shape = mobilenet_input_shape if use_transfer_learning_architecture else lfw_a_input_shape

    batch_size = 64
    max_trials = 150
    num_models = 10
    epochs = 50
    patience = 15
    min_delta = 0.03

    tf.random.set_seed(seed)
    np.random.seed(seed)

    if not run_bayes_search:
        hp = HyperParameters()
        hp.Fixed('learning_rate', 0.00001)
        hp.Fixed('dense_layer_size', 1024)
        hp.Fixed('enable_batch_normalization', False)
        hp.Fixed('bias_initializer', "zeros")
        hp.Fixed('conv2D_kernel_initializer', "he_normal")
        hp.Fixed('dense_kernel_initializer',  "he_normal")
        hp.Fixed('dropout_rate', 0.0)
        hp.Fixed('distance_metric', "abs")
        hp.Fixed('l2_regularizer', -1)
        hp.Fixed('optimizer', "rmsprop")
        max_trials = 1
        num_models = 1

    log_file_name = directory + "/" + project_name + "_log.txt"
    with open(log_file_name, "a") as h, contextlib.redirect_stdout(h):
        # preprocessing and dataset creation
        train_ds, val_ds, test_ds, one_shot_val_ds_list, one_shot_test_ds_list = make_dataset(
            images_directory=images_directory,
            resize_dim=input_shape,
            batch_size=batch_size,
            augment_training_dataset=augment_dataset,
            use_transfer_learning_architecture=use_transfer_learning_architecture,
            seed=seed)

        # configure bayesian optimization tuner
        tuner = BayesianOptimization(
            SiameseNeuralNetworkHyperModel(input_shape=input_shape,
                                           batch_size=batch_size,
                                           use_transfer_learning_architecture=use_transfer_learning_architecture,
                                           seed=seed),
            max_trials=max_trials,
            hyperparameters=hp,
            allow_new_entries=False,
            objective='val_binary_accuracy',
            seed=seed,
            num_initial_points=3,
            directory=directory,
            project_name=project_name,
            overwrite=overwrite)

        print()
        tuner.search_space_summary()
        print()

        # configure tensorboard callback and writer
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tf_log_dir)
        tf_writer = tf.summary.create_file_writer(file_path(tf_log_dir))
        tf_writer.set_as_default()

        # configure one shot learning accuracy test callback
        one_shot_accuracy_test_callback = OneShotLearningAccuracyTestCallback(one_shot_val_ds_list)

        # configure early stopping callback
        early_stop_callback = EarlyStopping(monitor='val_binary_accuracy',
                                            min_delta=min_delta,
                                            patience=patience,
                                            verbose=1,
                                            restore_best_weights=True)

        # perform bayesian search on the defined search space
        tuner.search(train_ds,
                     epochs=epochs,
                     validation_data=val_ds,
                     callbacks=[tensorboard_callback, early_stop_callback] if run_bayes_search else [tensorboard_callback, one_shot_accuracy_test_callback, early_stop_callback],
                     verbose=2)

        print()
        tuner.results_summary(num_models)

        print()
        best_trials = tuner.oracle.get_best_trials(num_models)
        best_trials_results = []
        for idx, model in enumerate(tuner.get_best_models(num_models=num_models)):
            # evaluate on the test set
            loss, accuracy = model.evaluate(test_ds, verbose=2)
            # check one shot learning on test set
            test_one_shot_accuracy_score = test_n_way(one_shot_test_ds_list, model)

            # evaluate on the validation set
            val_loss, val_accuracy = model.evaluate(val_ds, verbose=2)
            # check one shot learning on val set
            val_one_shot_accuracy_score = test_n_way(one_shot_val_ds_list, model)

            # evaluate on the train set
            train_loss, train_accuracy = model.evaluate(train_ds, verbose=2)

            hyperparameters = best_trials[idx].hyperparameters.values

            test_stats_and_scores = {"Trial ID": best_trials[idx].trial_id,
                                     "test loss": loss,
                                     "test accuracy": accuracy,
                                     "test one shot accuracy": test_one_shot_accuracy_score,
                                     "val loss": val_loss,
                                     "val accuracy": val_accuracy,
                                     "val one shot accuracy": val_one_shot_accuracy_score,
                                     "train loss": train_loss,
                                     "train accuracy": train_accuracy}
            test_stats_and_scores.update(hyperparameters)

            best_trials_results.append(test_stats_and_scores)

        # save statistics into a csv file
        csv_header = best_trials_results[0].keys()

        with open(csv_file_name, 'w') as csv_file:
            dict_writer = csv.DictWriter(csv_file, csv_header)
            dict_writer.writeheader()
            dict_writer.writerows(best_trials_results)


if __name__ == "__main__":
    main()
