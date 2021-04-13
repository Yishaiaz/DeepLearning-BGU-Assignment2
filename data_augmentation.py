import tensorflow as tf
import skimage as ski


def noise(image: tf.Tensor):
    return tf.convert_to_tensor(ski.util.random_noise(image.numpy(), mode='s&p'))


def rotation45(image: tf.Tensor):
    return tf.convert_to_tensor(ski.transform.rotate(image.numpy(), 45))


def center_crop(image: tf.Tensor):
    return tf.image.resize_with_crop_or_pad(tf.image.central_crop(image, 0.5), 250, 250)


def saturation(image: tf.Tensor):
    return tf.image.adjust_saturation(image, 3)


def noise_and_center_crop(image: tf.Tensor):
    return center_crop(noise(image))


def augment_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    noise_dataset = dataset.map(noise)
    rotation45_dataset = dataset.map(rotation45)
    center_crop_dataset = dataset.map(center_crop)
    saturation_dataset = dataset.map(saturation)
    noise_and_center_crop_dataset = dataset.map(noise_and_center_crop)

    augmented_dataset = dataset.concatenate(noise_dataset)
    augmented_dataset = augmented_dataset.concatenate(rotation45_dataset)
    augmented_dataset = augmented_dataset.concatenate(center_crop_dataset)
    augmented_dataset = augmented_dataset.concatenate(saturation_dataset)
    augmented_dataset = augmented_dataset.concatenate(noise_and_center_crop_dataset)

    return augmented_dataset
