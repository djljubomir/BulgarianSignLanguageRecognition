import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

model = tf.keras.models.load_model("SLR-8E-CNN.model")

sign_names = ["а", "б", "в", "г", "д", "е", "ж", "з", "и", "й", "к", "л", "м", "н", "о", "п", "р", "с", "т", "у", "ф",
              "х", "ц", "ч", "ш", "щ", "ъ", "ь", "ю", "я", "интервал", "изтриване", "нищо"]

path_to_directory = os.path.join("Test/", "")
test_data = []
for test_sign_image in os.listdir(path_to_directory):
    test_sign_images_array = cv2.imread(os.path.join(path_to_directory, test_sign_image), cv2.IMREAD_GRAYSCALE)
    test_sign_images_array = cv2.flip(test_sign_images_array, 1)
    test_sign_images_array = cv2.resize(test_sign_images_array, (256, 256))
    # cv2.imshow("images", test_sign_images_array)
    # cv2.waitKey()
    test_sign_images_array = test_sign_images_array / 255

    test_data.append(test_sign_images_array)  # add thi
# print(test_data)
# prediction = model.predict([prepare(test_data[0])])

for index in range(-1, len(test_data)):
    predictions_array = model.predict([test_data[index].reshape(-1, 256, 256, 1)])
    print(predictions_array)

    prediction = np.argmax(predictions_array[0])
    print(sign_names[prediction])

    # plt.imshow(test_data[index], cmap='gray')
    # plt.title(test_data[3])
    # plt.show()
