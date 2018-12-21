import matplotlib
matplotlib.use('Agg')

import numpy as np
import os
import Augmentor

from keras import applications
from keras.utils import Sequence
from keras.layers import Dense, Activation
from keras.preprocessing.image import load_img, img_to_array
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from sklearn.utils import shuffle
from datetime import datetime

from generators import Generator
from models import Model01

######################
# Settings
######################

batch_size = 64

######################
# Set up generators
######################

# Training Generator

train_dir = '/media/kashgar/data/pnn_training/p256/train/'
se_x = os.listdir(os.path.join(train_dir, 'se'))
se_x = [os.path.join(train_dir, 'se', x) for x in se_x]
se_y = [1] * len(se_x)
nse_x = os.listdir(os.path.join(train_dir, 'nse'))
nse_x = [os.path.join(train_dir, 'nse', x) for x in nse_x]
nse_y = [0] * len(nse_x)

se_x, se_y = shuffle(se_x, se_y)
nse_x, nse_y = shuffle(nse_x, nse_y)

n_se = len(se_x)
n_nse = len(nse_x)
n = np.min([n_se, n_nse])

se_x, se_y, nse_x, nse_y = se_x[:n], se_y[:n], nse_x[:n], nse_y[:n]

x = se_x + nse_x
y = se_y + nse_y
x, y = shuffle(x, y)

p = Augmentor.Pipeline()
p.flip_left_right(probability=0.5)
p.rotate(probability=1.0, max_left_rotation=25, max_right_rotation=25)
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
p.random_brightness(probability=0.5, min_factor=0.6, max_factor=1.5)
p.random_color(probability=0.5, min_factor=0.75, max_factor=0.9)
p.random_contrast(probability=0.5, min_factor=0.75, max_factor=0.9)
p.random_erasing(probability=0.5, rectangle_area=0.5)

traingen = Generator(x, y, batch_size, p)

# Validation Generator

dir_ = '/media/kashgar/data/pnn_training/p256/validation/'
se_x = os.listdir(os.path.join(dir_, 'se'))
se_x = [os.path.join(dir_, 'se', x) for x in se_x]
se_y = [1] * len(se_x)
nse_x = os.listdir(os.path.join(dir_, 'nse'))
nse_x = [os.path.join(dir_, 'nse', x) for x in nse_x]
nse_y = [0] * len(nse_x)

se_x, se_y = shuffle(se_x, se_y)
nse_x, nse_y = shuffle(nse_x, nse_y)

n_se = len(se_x)
n_nse = len(nse_x)
n = np.min([n_se, n_nse])

se_x, se_y, nse_x, nse_y = se_x[:n], se_y[:n], nse_x[:n], nse_y[:n]

x = se_x + nse_x
y = se_y + nse_y
x, y = shuffle(x, y)

validgen = Generator(x, y, batch_size, pipeline=None)

######################
# Load Model
######################

model = Model01()
opt = Adam(lr=0.0001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

######################
# Training
######################

now = datetime.now().strftime('%Y%m%d%H%M%S')
early_stopping = EarlyStopping(monitor='val_loss', patience=25)
checkpoint = ModelCheckpoint('logs/{now}/'.format(now=now) + 'epoch_{epoch:04d}-{val_loss:.4f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
tb = TensorBoard(log_dir='logs/{now}'.format(now=now), batch_size=batch_size, write_grads=True, write_images=True)
model.fit_generator(
    traingen,
    epochs = 10000,
    validation_data = validgen,
    callbacks=[early_stopping, checkpoint, tb],
    use_multiprocessing=True,
    shuffle=True,
    workers=10,
    max_queue_size=50)
