from tensorboard.plugins.hparams import api as hp

# configure hyper-parameters grid-search
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['sgd']))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.1, 0.01, 0.001, 0.0001]))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32, 64, 128, 512]))
HP_DENSE_UNITS_NUM = hp.HParam('dense_units_num', hp.Discrete([4096, 2048, 1024]))