# coinsDetection

Coin Recognize Software for Computer Vision course

## Folder Structure

    .
    ├── CMakeList.txt
    ├── model-final-*.h5               	    # Trained model of the ANN
    ├── model-final-*.json            	    # Trained model of the ANN
    ├── src                 	  	    # Source files
    |   └── coinRecognize.cpp			# Principal Class
    |   └── main.cpp				# Main class
    |   └── networkTrainer.py			# Trainer for the net ##RUN IT ON GOOGLE COLAB
    |   └── tester.py				# Tester for prediction
    ├── include                 	    # Header file
    |   └── coinRecognize.h 			# Header for main class
    ├── dataset                 	    # Dataset for training divided in class
    |   └── 1e	
    |   └── 2e
    |   └── 20c
    |   └── 50c
    |   └── unknown
    ├── doc                  		    # documentation folder (Doxygen)
    |   └── ...
    └── images              	 	    # Classified example images
        ├── pic			  	  	# Sample images for testing
        └── predict               	 
            └── coins				# Single images of coin
            └── prediction.csv			# CSV file with predictions
	
## Use

For *compiling* do the following:

	mkdir build
	cd build
	cmake ..
	make

And then *run* with

	./coinRecognize ["path_of_image"]

Use passing the path of the image that you like to test or change one of the first line of the code.
If you don't pass an argument the software pick a sample images from the pic folder.


## About 

@author Denis Donadel <mailto:denis.donadel@studenti.unipd.it>
@date 6 July 2019
@version 1.0
@since 1.0
