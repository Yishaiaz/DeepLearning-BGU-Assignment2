import csv

import tensorflow as tf
from kerastuner import BayesianOptimization
from tensorflow.python.keras.callbacks import EarlyStopping

from config import hp
from preprocessing_utils import make_dataset
from siamese_network import SiameseNeuralNetworkHyperModel
from utils import file_path, test_n_way, OneShotLearningAccuracyTestCallback


# Global configuration
seed = 0
lfw_a_input_shape = (250, 250, 1)
images_directory = 'lfw2Data/lfw2'
augment_dataset = True
project_name = "batch_32_with_data_augmentation"
directory = "tuner_results"
csv_file_name = project_name + ".csv"
batch_size = 32
max_trials = 2
num_models = 2
epochs = 2
patience = 10
min_delta = 1


def main():
    # preprocessing and dataset creation
    train_ds, val_ds, test_ds, one_shot_val_ds_list, one_shot_test_ds_list = make_dataset(
        images_directory=images_directory,
        batch_size=batch_size,
        augment_training_dataset=augment_dataset,
        seed=seed)

    # configure bayesian optimization tuner
    tuner = BayesianOptimization(
        SiameseNeuralNetworkHyperModel(input_shape=lfw_a_input_shape, batch_size=batch_size, seed=seed),
        max_trials=max_trials,
        hyperparameters=hp,
        allow_new_entries=False,
        objective='val_binary_accuracy',
        seed=seed,
        directory=directory,
        project_name=project_name,
        overwrite=False)

    print()
    tuner.search_space_summary()
    print()

    # configure tensorboard callback and writer
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./tf_logs/tuner/" + project_name)
    tf_writer = tf.summary.create_file_writer(file_path("./tf_logs/tuner/" + project_name))
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
                 callbacks=[tensorboard_callback, one_shot_accuracy_test_callback, early_stop_callback])

    print()
    tuner.results_summary(num_models)

    print()
    best_trials = tuner.oracle.get_best_trials(num_models)
    best_trials_results = []
    for idx, model in enumerate(tuner.get_best_models(num_models=num_models)):
        # evaluate on the test set
        loss, accuracy = model.evaluate(test_ds)
        # check one shot learning on test set
        test_one_shot_accuracy_score = test_n_way(one_shot_test_ds_list, model)

        # evaluate on the validation set
        val_loss, val_accuracy = model.evaluate(val_ds)
        # check one shot learning on val set
        val_one_shot_accuracy_score = test_n_way(one_shot_val_ds_list, model)

        # evaluate on the train set
        train_loss, train_accuracy = model.evaluate(train_ds)

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
