import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

sign_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                54, 55, 56, 57, 58, 59, 60, 61]

sign_categories = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "Interval", "L1", "L2", "L3", "L4", "L5", "L6",
                   "L7", "L8", "L9", "L10", "L11", "L12", "L13", "L14", "L15", "L16", "L17", "L18", "L19", "L20", "L21",
                   "L22", "L23", "L24", "L25", "L26", "L27", "L28", "L29", "L30", "Nothing", "W1", "W2", "W3", "W4",
                   "W5", "W6", "W7", "W8", "W9", "W10", "W11", "W12", "W13", "W14", "W15", "W16", "W17", "W18", "W19",
                   "W20"]

data = []
for i in range(0, len(sign_categories), 1):
# for i in range(0, 1, 1):
    path_to_directory = os.path.join("Dataset/", sign_categories[i])
    sign_name = sign_indexes[i]
    for sign_image in os.listdir(path_to_directory):
        sign_images_array = cv2.imread(os.path.join(path_to_directory, sign_image), cv2.IMREAD_GRAYSCALE)
        # sign_images_array = cv2.flip(sign_images_array, 1)
        print(i)
        sign_images_array = cv2.resize(sign_images_array, (128, 128))  # resize to normalize data size
        data.append([sign_images_array, sign_name])  # add thi
        # cv2.imshow("Image", sign_images_array)
        # cv2.waitKey()

print(len(data))

random.shuffle(data)
for sample in data[:50]:
    print(sample[1])

X = []
y = []

for image, sign_name in data:
    X.append(image)
    y.append(sign_name)

# X = np.array(X).reshape(-1, 128, 128, 1)
# np.save('sign_images_test.npy', X)
np.save('sign_images_128x128_Full_kNN.npy', X)

y = np.array(y)
# np.save('sign_indexes_test.npy', y)
np.save('sign_indexes_128x128_Full_kNN.npy', y)




