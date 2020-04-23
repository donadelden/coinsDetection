######################################################
# Coin Recognize Software for Computer Vision course #
# 													 #
# Donadel Denis <denis.donadel@studenti.unipd.it> 	 #
# 													 #
# !!! THIS CODE IS THINKED TO RUN ON GOOGLE COLAB !! #
# Trainer for the NN								 #
# 6 July 2019										 #
# version 1.0 - since 1.0							 #
######################################################

# mount Google Drive as storage
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)


# import classifier
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import SGD, rmsprop
from keras.models import Model

# set the order of dim images
from keras import backend
backend.set_image_dim_ordering('th')

# import other stuff
import os
import datetime

#transfer learning
#from keras.applications.mobilenet_v2 import preprocess_input # another tested net, not so good as inceptionV3
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

# constants
## path with data used in training
TRAIN_PATH = '/content/gdrive/My Drive/Colab Notebooks/dataset-total'
## batch size
BATCH_SIZE = 32
## images dimensions
HEIGHT = WIDTH = 224
## epochs for the train
EPOCHS = 7


def create_model_keras(l_num):
	model = Sequential()
	model.add(Conv2D(96, (11, 11), activation='relu',
					 input_shape=(3, INPUT_SHAPE, INPUT_SHAPE)))
	model.add(Conv2D(256, (5, 5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(384, (3, 3), activation='relu'))
	model.add(Conv2D(384, (3, 3), activation='relu'))
	model.add(Conv2D(384, (3, 3), activation='relu'))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(2048, activation='relu'))
	model.add(Dense(2048, activation='relu'))
	model.add(Dense(2048, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(l_num, activation='softmax'))
	return model


if __name__ == '__main__':

	labelsNum = sum(os.path.isdir(os.path.join(TRAIN_PATH,i)) for i in os.listdir(TRAIN_PATH))
	print ("Number of labels: " + str(labelsNum))
	LABELS = [i for i in os.listdir(TRAIN_PATH) if os.path.isdir(os.path.join(TRAIN_PATH,i))]
	LABELS.sort()
	print("Labels: " + str(LABELS))

	#imports the Inception model discarding the last 1000 neuron layer
	base_model=InceptionV3(weights='imagenet', include_top=False)

	x=base_model.output
	x=GlobalAveragePooling2D()(x)
	# add fully connected layers so that the model can learn more complex functions and classify for better results
	x=Dense(1024,activation='relu')(x)
	x=Dense(512,activation='relu')(x)
	#final layer with softmax activation
	preds=Dense(labelsNum,activation='softmax')(x)

	# create the model based on Inception V3
	model=Model(inputs=base_model.input, outputs=preds)

	# print layers # DEBUG
	#print("Network: \n")
	#for i,layer in enumerate(model.layers):
	#  print(i,layer.name)

	# train only the few last layer
	for layer in model.layers[:20]:
		layer.trainable=False
	for layer in model.layers[20:]:
		layer.trainable=True

	#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) # better rms!
	rms = rmsprop(lr=0.0001, decay=1e-6)

	# use Inception V3 preprocessing included in our dependencies with some data augmentation techniques
	train_datagen = ImageDataGenerator(
		rotation_range=90,
		width_shift_range=0.3,
		height_shift_range=0.3,
		zoom_range=0.2,
		preprocessing_function=preprocess_input,
		validation_split=0.2) # set the size of validation data

	# this is a generator that will read pictures found in
	# subfolers of path, and indefinitely generate
	# batches of augmented image data
	train_generator = train_datagen.flow_from_directory(
		TRAIN_PATH,
		target_size=(HEIGHT, WIDTH),
		batch_size=BATCH_SIZE,
		class_mode='categorical',
		subset='training') # set as training data

	validation_generator = train_datagen.flow_from_directory(
		TRAIN_PATH,
		target_size=(HEIGHT, WIDTH),
		batch_size=BATCH_SIZE,
		class_mode='categorical',
		subset='validation') # set as validation data

	# print shape of trining images
	sampleBatch = next(train_generator)
	print('Train img shape: ' + str(sampleBatch[0].shape))

	# compile the model
	model.compile(optimizer=rms,
				  loss='categorical_crossentropy',
				  metrics=['accuracy'])

	# train the net
	model.fit_generator(
		train_generator,
		steps_per_epoch=1000 // BATCH_SIZE,
		validation_data=validation_generator,
		validation_steps=50 // BATCH_SIZE,
		epochs=	EPOCHS)

	# saving of model
	model_filename = 'model-' + str(EPOCHS) + 'epochs-' + str(datetime.datetime.now().isoformat())

	# serialize model to JSON and save on Google Drive
	model_json = model.to_json()
	with open('/content/gdrive/My Drive/Colab Notebooks/' + model_filename + '.json', "w") as json_file:
		json_file.write(model_json)

	# serialize weights to HDF5 and save on Google Drive
	model.save_weights('/content/gdrive/My Drive/Colab Notebooks/' + model_filename + '.h5')
	print("Saved model to disk as " + model_filename)
