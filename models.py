from keras.layers import Input, Conv2D, Activation, MaxPooling2D, Flatten, Dense, Concatenate, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.models import Model

def Model01(size=(256, 256, 3)):
	inputTensor = Input(size)
	# block 1
	x = Conv2D(64, (3, 3),
		activation='relu',
		padding='same',
		name='block1_conv1')(inputTensor)
	x = Conv2D(64, (3, 3),
		activation='relu',
		padding='same',
		name='block1_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool1')(x)
	# rf is 6 x 6
	# rep is 112 x 112 x 64

	# block 2
	x = Conv2D(128, (3, 3), 
		activation='relu',
		padding='same',
		name='block2_conv1')(x)
	x = Conv2D(128, (3, 3),
		activation='relu',
		padding='same',
		name='block2_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
	# rf is 16 x 16
	# rep is 56 x 56 x 128

	# block 3
	x = Conv2D(256, (3, 3),
		activation='relu',
		padding='same',
		name='block3_conv1')(x)
	x = Conv2D(256, (3, 3),
		activation='relu',
		padding='same',
		name='block3_conv2')(x)
	x = Conv2D(256, (3, 3),
		activation='relu',
		padding='same',
		name='block3_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
	# rf is 44 x 44
	# rep is 28 x 28 x 256

	# block 4
	x = Conv2D(512, (3, 3),
		activation='relu',
		padding='same',
		name='block4_conv1')(x)
	x = Conv2D(512, (3, 3),
		activation='relu',
		padding='same',
		name='block4_conv2')(x)
	# TRUNCATE HERE
	# rf is 76 x 76
	# rep is 28 x 28 x 512 ()

	# HEAD

	# 1-D block
	x = Conv2D(512, (1, 1), 
		activation='relu',
		padding='same',
		name='head_conv1')(x)
	x = Conv2D(512, (1, 1), 
		activation='relu',
		padding='same',
		name='head_conv2')(x)
	x = Conv2D(512, (1, 1), 
		activation='relu',
		padding='same',
		name='head_conv3')(x)

	# GAP
	x = GlobalAveragePooling2D(name='gap')(x)

	# Predict
	x = Dense(1, activation='sigmoid', name='fc_1')(x)

	model = Model(inputTensor, x, name='model01')

	return model