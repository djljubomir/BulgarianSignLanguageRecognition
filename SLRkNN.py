from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_validate
import numpy as np
import cv2
import imutils
import pickle
from sklearn.utils import shuffle


def extract_color_histogram(image, bins=(8, 8, 8)):
    # extract a 3D color histogram from the HSV color space using
    # the supplied number of `bins` per channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
        [0, 180, 0, 256, 0, 256])
    # handle normalizing the histogram if we are using OpenCV 2.4.X
    if imutils.is_cv2():
        hist = cv2.normalize(hist)
    # otherwise, perform "in place" normalization in OpenCV 3 (I
    # personally hate the way this is done
    else:
        cv2.normalize(hist, hist)
    # return the flattened histogram as the feature vector
    return hist.flatten()


sign_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                54, 55, 56, 57, 58, 59, 60, 61]

sign_categories = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "Interval", "L1", "L2", "L3", "L4", "L5", "L6",
                   "L7", "L8", "L9", "L10", "L11", "L12", "L13", "L14", "L15", "L16", "L17", "L18", "L19", "L20", "L21",
                   "L22", "L23", "L24", "L25", "L26", "L27", "L28", "L29", "L30", "Nothing", "W1", "W2", "W3", "W4",
                   "W5", "W6", "W7", "W8", "W9", "W10", "W11", "W12", "W13", "W14", "W15", "W16", "W17", "W18", "W19",
                   "W20"]

X = np.load('sign_images_128x128_Full_kNN.npy')
y = np.load('sign_indexes_128x128_Full_kNN.npy')

# X = np.load('sign_images_test.npy')
# y = np.load('sign_indexes_test.npy')

# print(X)
# X.flatten()
# X = extract_color_histogram(X, bins=(8, 8, 8))

X = X/255.0
X, y = shuffle(X, y)
print(X.shape)
X = np.array(X).reshape(15500, 16384)

# for i in X:
#     X[i].flatten()

# for i in X:
#     X[i] = extract_color_histogram(X[i])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

# Its important to use binary mode
knn_clf = open('SLR-128x128-Full-kNN_clf', 'wb')
# source, destination
pickle.dump(clf, knn_clf, protocol=4)

# load the model from disk
# loaded_model = pickle.load(open('SLR-128x128-GrayScale_with_gen_background_kNN_clf', 'rb'))
# result = loaded_model.predict(X_test)

