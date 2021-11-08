from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import imutils
import pickle
from sklearn.utils import shuffle

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

X = X/255.0

X, y = shuffle(X, y)
# print(X.shape[1:])

print(X.shape)
X = np.array(X).reshape(15500, 16384)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
model = SVC(C=1, kernel='poly', gamma='auto')
model.fit(X_train, y_train)

# prediction = model.predict(X_test, y_test)
accuracy = model.score(X_test, y_test)
print(accuracy)

# Its important to use binary mode
svm_clf = open('SLR-128x128-Full_SVM_clf', 'wb')
# source, destination
pickle.dump(model, svm_clf, protocol=4)

# load the model from disk
# loaded_model = pickle.load(open('SLR-128x128-GrayScale_with_gen_background_SVM_clf', 'rb'))
# result = loaded_model.predict(X_test)