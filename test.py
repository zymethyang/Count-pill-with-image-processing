#import lib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import math
import numpy as np
import argparse
import glob
import cv2

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

def predictMaterial(roi):
    # calculate feature vector for region of interest
    hist = calcHistogram(roi)

    # predict material type
    s = clf.predict([hist])

    # return predicted material type
    return Material[int(s)]

# define Enum class
class Enum(tuple): __getattr__ = tuple.index


# Enumerate material types for use in classifier
Material = Enum(('Back', 'Pill'))

# locate sample image files
sample_images_pill = glob.glob("sample_images/pill/*")
sample_images_none = glob.glob("sample_images/none/*")

# define training data and labels
X = []
y = []

# compute and store training data and labels
for i in sample_images_pill:
    X.append(calcHistFromFile(i))
    y.append(Material.Pill)
for i in sample_images_none:
    X.append(calcHistFromFile(i))
    y.append(Material.Back)

#MLP layer
# instantiate classifier
# Multi-layer Perceptron
# score: 0.974137931034
clf = MLPClassifier(solver="lbfgs")

# split samples into training and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2)

# train and score classifier
clf.fit(X_train, y_train)
score = int(clf.score(X_test, y_test) * 100)
print("Classifier mean accuracy: ", score)

cap = cv2.VideoCapture(1)


while(True):
    # Capture frame-by-frame
    ret, image = cap.read()
    d = 200 / image.shape[1]
    dim = (200, int(image.shape[0] * d))
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    cv2.imshow("Output", np.hstack([blurred]))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()