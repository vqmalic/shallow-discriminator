import matplotlib
matplotlib.use("agg")

import matplotlib.pyplot as plt

import os
import numpy as np
import pickle as pkl

from keras.models import load_model
from utils import path_to_array
from tqdm import tqdm
from scipy.spatial.distance import cdist
from PIL import Image

import keras.backend as K

RET_PATH = "/home/vqmalic/projects/pnn_jigsaw/retrieval_images_full/"
TEST_PATH = "/media/kashgar/data/pnn_training/p256/train/se/"
N = 5000000
keep = 100
batch_size = 128
n_batches = int(np.ceil(N/batch_size))

model = load_model("/home/vqmalic/projects/shallow-discriminator/logs/20181224011750/epoch_0026-0.0352.hdf5")

##########################
# function for rep layer
##########################

l = model.layers[-2]
oup = l.output
f = K.function([model.input], [oup])

##########################
# query reps
##########################

retpaths = os.listdir(RET_PATH)
retpaths = [os.path.join(RET_PATH, x) for x in retpaths]
retpaths = sorted(retpaths)

input_ = []

for path in retpaths:
    arr = path_to_array(path)
    input_.append(arr)

input_ = np.array(input_)
reps = f([input_])[0]

##########################
# perform search
##########################

nearest = [[] for i in range(len(retpaths))]

test_paths = os.listdir(TEST_PATH)
test_paths = [os.path.join(TEST_PATH, x) for x in test_paths]

for i in tqdm(range(n_batches)):
    these_imgs = np.random.choice(test_paths, size=batch_size, replace=False)
    to_nn = []
    for img in these_imgs:
        arr = path_to_array(img)
        to_nn.append(arr)
    to_nn = np.array(to_nn)
    these_reps = f([to_nn])[0]

    d = cdist(reps, these_reps)

    for cand_idx in range(d.shape[0]):
        for tar_idx in range(d.shape[1]):
            this_d = d[cand_idx, tar_idx]
            if len(nearest[cand_idx]) < keep:
                nearest[cand_idx].append((these_imgs[tar_idx], offsets[tar_idx], this_d))
            else:
                current = nearest[cand_idx]
                current_vals = np.array([x[-1] for x in current])
                max_where = np.argmax(current_vals)
                max_val = np.max(current_vals)
                if this_d < max_val:
                    current.pop(max_where)
                    current.append((these_imgs[tar_idx], offsets[tar_idx], this_d))