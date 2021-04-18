from kerastuner import HyperParameters
from tensorboard.plugins.hparams import api as hp

# TODO remove
# configure hyper-parameters search
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['sgd']))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.1, 0.01, 0.001, 0.0001]))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([8, 32, 64, 128, 512]))
HP_DENSE_UNITS_NUM = hp.HParam('dense_units_num', hp.Discrete([4096, 2048, 1024]))


hp = HyperParameters()
hp.Choice('learning_rate', [1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
hp.Choice('dense_layer_size', [256, 512, 1024, 2048, 4096])
hp.Boolean('enable_batch_normalization')
hp.Choice('bias_initializer', ["default", "zeros"])
hp.Choice('conv2D_kernel_initializer', ["default", "he_normal", "glorot_normal"])
hp.Choice('dense_kernel_initializer', ["default", "he_normal", "glorot_normal"])
hp.Choice('dropout_rate', [0.0, 0.2, 0.3, 0.5, 0.7])
hp.Choice('distance_metric', ["abs", "euclidean_distance"])
hp.Choice('l2_regularizer', [-1.0, 0.001, 0.005, 0.1, 0.2, 0.3, 0.5, 0.9])
hp.Choice('optimizer', ["adam", "sgd", "rmsprop"])
