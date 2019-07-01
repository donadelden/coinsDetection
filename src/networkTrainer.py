# import classifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import SGD, rmsprop
from keras import backend

backend.set_image_dim_ordering('th') # set the order of dim images

# import other stuff
import glob
import cv2
import numpy as np
import keras
import os
import datetime

INPUT_SHAPE = 150
LABELS = 5
TRAIN_PATH = '../coins-dataset/classified/total/preprocessed/'
BATCH_SIZE = 32
HEIGHT = WIDTH = 150
EPOCHS = 50


def planImage(file):
	img = cv2.imread(file)
	return img#.flatten()

def create_model_keras(l_num):
    model = Sequential()
    model.add(Conv2D(64, (5, 5), activation='relu',
		input_shape=(3, INPUT_SHAPE, INPUT_SHAPE)))
    model.add(Conv2D(64, (5, 5), activation='relu')) #maybe useless
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(l_num, activation='softmax'))
    return model



#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
rms = rmsprop(lr=0.0001, decay=1e-6)
model = create_model_keras(l_num = LABELS)
model.compile(loss='categorical_crossentropy',
			  optimizer=rms,
			  metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
		rescale=1./255,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		validation_split=0.2) # set validation split

# todo: reactivate this
# this is the augmentation configuration we will use for testing:
# only rescaling
# test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of path, and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
	TRAIN_PATH,  # this is the target directory
	target_size=(HEIGHT, WIDTH),  # all images will be resized to 150x150
	batch_size=BATCH_SIZE,
	class_mode='categorical',
	subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(HEIGHT, WIDTH),  # all images will be resized to 150x150
	batch_size=BATCH_SIZE,
	class_mode='categorical',
    subset='validation') # set as validation data

# DEBUG images shape
# train_generator is DirectoryIterator yielding tuples of (x, y) where x is a
# numpy array containing a batch of images with shape
# (batch_size, *target_size, channels) and y is a numpy array of corresponding labels

sample_batch = next(train_generator)
print('Train img shape: ' + str(sample_batch[0].shape))

# todo: reactivate this
# this is a similar generator, for validation data
# validation_generator = test_datagen.flow_from_directory(
#        TRAIN_PATH,
#        target_size=(150, 150),
#        batch_size=BATCH_SIZE,
#        class_mode='categorical')

# TRAIN
model.fit_generator(
		train_generator,
		steps_per_epoch=1000 // BATCH_SIZE,
		validation_data=validation_generator,
		validation_steps=50 // BATCH_SIZE,
		epochs=	EPOCHS)

# saving of model
model_filename = 'model' + str(datetime.datetime.now().isoformat())

# serialize model to JSON
model_json = model.to_json()
with open(model_filename + '.json', "w") as json_file:
	json_file.write(model_json)

# serialize weights to HDF5
model.save_weights(model_filename + '.h5')  # always save your weights after training or during training
print("Saved model to disk as " + model_filename)
