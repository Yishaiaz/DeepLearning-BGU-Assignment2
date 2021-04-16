import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


all_images_path = 'lfw2Data/lfw2'

main_directory_content = [os.sep.join([all_images_path, x]) for x in list(filter(lambda x: not str(x).lower().find('ds_store')!=-1, os.listdir(all_images_path)))]


def plot_distribution_num_of_images():
    def get_num_of_images_for_person() -> dict:
        names_to_num_of_images = {}
        for person_dir in main_directory_content:
            names_to_num_of_images[person_dir] = len(os.listdir(person_dir))
        return names_to_num_of_images
    fig, ax = plt.subplots()
    names_to_num_of_images_dict = get_num_of_images_for_person()
    num_of_images = np.array(list(names_to_num_of_images_dict.values()))
    # min max normalization
    norm_num_of_images = (num_of_images - num_of_images.min())/(num_of_images.max() - num_of_images.min())
    ax.hist(norm_num_of_images)
    # ax.set_xticks(range(0, 1000, 100))
    ax.set_title('total number of images {}\n total number of people {}'.format(sum(names_to_num_of_images_dict.values()), len(names_to_num_of_images_dict.keys())))



if __name__ == '__main__':
    print('EDA start')
    plot_distribution_num_of_images()
    plt.show()
    print('EDA done')
