#   -*- coding: utf-8 -*-
#
#   cfg.py
#   
#   Developed by Tianyi Liu on 2020-05-26 as tianyi
#   Copyright (c) 2020. All Rights Reserved.

"""

"""


# Seeds
NUMPY_RAND_SEED = 123
TORCH_RAND_SEED = 123

# Model
DROPOUT_PROB = 0.2

# Preproc
LOG_NORMALIZE_SCALE = 10000

# Caches
CACHE_WRITE = True
CACHE_DIR = "./cache"
CACHE_DATA_NAME = "cache.pkl"
CACHE_DATA_FULL_PATH = CACHE_DIR + "/" + CACHE_DATA_NAME

# Read data
LABEL_COL = "proc"
BATCH_ID = "batch_id"

# Learning rate decay
LR_DECAY_EPOCH = 20
LR_DECAY_GAMMA = 1e-1
LR_DECAY_MIN = 1e-5

# Pars
PCA_DIM = 50
PRINT_STEP = 30
VISUL_EPOCH = 50

# Visualization
VISUL_DIR = "./results/visualization"
TSNE_JOBS = 4

# Logging
LOG_PATH = "./results/log"

