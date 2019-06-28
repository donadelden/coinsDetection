# import classifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# import other stuff
import glob
import cv2
import numpy as np

def calcHistogram(img):
    # create mask
    m = np.zeros(img.shape[:2], dtype="uint8")
    (w, h) = (int(img.shape[1] / 2), int(img.shape[0] / 2))
    cv2.circle(m, (w, h), 60, 255, -1)

    # calcHist expects a list of images, color channels, mask, bins, ranges
    h = cv2.calcHist([img], [0, 1, 2], m, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    # return normalized "flattened" histogram
    return cv2.normalize(h, h).flatten()


def calcHistFromFile(file):
    img = cv2.imread(file)
    return calcHistogram(img)


# locate sample image files
path = "../coins-dataset/classified/total/"
sample_images_20c = glob.glob(path + "20c/*")
sample_images_50c = glob.glob(path + "50c/*")
sample_images_e1 = glob.glob(path + "1e/*")
sample_images_e2 = glob.glob(path + "2e/*")
print("len:" + str(len(sample_images_20c)) + " " + str(len(sample_images_50c))
		+ " " + str(len(sample_images_e1)) + " " + str(len(sample_images_e2)));

# define training data and labels
X = []
y = []

# compute and store training data and labels
for i in sample_images_20c:
    X.append(calcHistFromFile(i))
    y.append(20)
for i in sample_images_50c:
    X.append(calcHistFromFile(i))
    y.append(50)
for i in sample_images_e1:
    X.append(calcHistFromFile(i))
    y.append(1)
for i in sample_images_e2:
    X.append(calcHistFromFile(i))
    y.append(2)

# instantiate classifier
# Multi-layer Perceptron
# score: 0.974137931034
clf = MLPClassifier(solver="lbfgs")

# split samples into training and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2)

# train and score classifier
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print("Classifier mean accuracy: ", score)
