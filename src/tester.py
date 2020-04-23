######################################################
# Coin Recognize Software for Computer Vision course #
# 													 #
# Donadel Denis <denis.donadel@studenti.unipd.it> 	 #
# 													 #
# Tester for the NN								     #
# 6 July 2019										 #
# version 1.0 - since 1.0							 #
######################################################


from keras import backend as K
K.set_image_dim_ordering('th') # set the order of dim images

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.inception_v3 import preprocess_input

import os
import csv

## model filename
MODEL_FILENAME = "../model-final-003-7-2019-07-06T16_50_52.636267"
## path with data used in training (need only the subdirectories for the categories)
TRAIN_PATH = "../dataset/"
## path with images for make prediction
PREDICT_PATH = "../images/predict/"
## size for the image resize
HEIGHT = WIDTH = 224

## listing labels
LABELS = [i for i in os.listdir(TRAIN_PATH) if os.path.isdir(os.path.join(TRAIN_PATH,i))]
LABELS.sort()
print("Labels: " + str(LABELS))
if __name__ == '__main__':
    # preprocess images as in train
    predictDatagen = ImageDataGenerator(preprocessing_function = preprocess_input) #included in our dependencies

    predictGenerator = predictDatagen.flow_from_directory(
        PREDICT_PATH,  # target directory
        target_size=(WIDTH, HEIGHT),  # resizing
        batch_size=1, # bach of one image
        class_mode='categorical', # we need a categorical output
        shuffle=False) # no shuffle!!!

    # load pretrained model
    json_file = open(MODEL_FILENAME + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(MODEL_FILENAME + ".h5")
    print("Model correctly loaded")

    # generate prediction
    samples = len(predictGenerator.filenames)
    predictions = model.predict_generator(predictGenerator,
                                          steps=samples,
                                          verbose=0) # only in DEBUG set to 1


    print("Predictions: " + str(predictions))

    # print predicted label
    predIndex = predictions.argmax(axis=-1)
    predLabel = [LABELS[i] for i in predIndex]
    print("Predicted labels: " + str(predLabel))

    # save in a .csv the output such that it can be opended by C++ program
    with open('../images/predict/predictions.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(predLabel)
    csvFile.close()
