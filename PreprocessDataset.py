import numpy as np
import os
import cv2
from random import randint


sign_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                54, 55, 56, 57, 58, 59, 60, 61]

sign_categories = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "Interval", "L1", "L2", "L3", "L4", "L5", "L6",
                   "L7", "L8", "L9", "L10", "L11", "L12", "L13", "L14", "L15", "L16", "L17", "L18", "L19", "L20", "L21",
                   "L22", "L23", "L24", "L25", "L26", "L27", "L28", "L29", "L30", "Nothing", "W1", "W2", "W3", "W4",
                   "W5", "W6", "W7", "W8", "W9", "W10", "W11", "W12", "W13", "W14", "W15", "W16", "W17", "W18", "W19",
                   "W20"]


# def genHumanSkinColor():
#     daylight = randint(0, 1)
#     print(daylight)
#     if daylight == 0:
#         R = randint(221, 255)
#         print(R)
#         G = randint(max(R - 15, 211), min(R + 15, 255))
#         print(G)
#         B = randint(171, (min(R, G) - 1))
#         print(B)
#     else:
#         R = randint(96, 255)
#         print(R)
#         G = randint(41, R - 16)
#         print(G)
#         B = randint(21, R - 1)
#         print(B)
#     return [B, G, R]
#
#
# def genBackground():
#     R = randint(0, 255)
#     G = randint(0, 255)
#     B = randint(0, 255)
#     return [B, G, R]


def vectorized_form(img):
    B, G, R = [img[:,:,x] for x in range(3)]
    return (R == 255) & (G == 255) & (B == 255)


data = []
# for i in range(0, len(sign_categories), 1):
for i in range(0, -1, 1):
    path_to_directory = os.path.join("Dataset/", sign_categories[i])
    sign_name = sign_indexes[i]
    for sign_image in os.listdir(path_to_directory):
        sign_images_array = cv2.imread(os.path.join(path_to_directory, sign_image), cv2.IMREAD_COLOR)
        # human_skin_color = genHumanSkinColor()

        hand_arr = vectorized_form(sign_images_array)
        non_mask_pix = hand_arr != 0  # select everything that is not mask_value
        mask_pix = hand_arr == 0
        # sign_images_array[non_mask_pix] = human_skin_color
        # sign_images_array[mask_pix] = genBackground()
        cv2.imwrite(os.path.join(path_to_directory, sign_image), sign_images_array)

        # for height in range(0, 600, 1):
        #     for width in range(0, 700, 1):
        #         # print(sign_images_array[height][width])
        #         if sign_images_array[height][width][0] != 255:
        #             sign_images_array[height][width] = genBackground()
        #         else:
        #             sign_images_array[height][width] = human_skin_color
        # cv2.imwrite(os.path.join(path_to_directory, sign_image), sign_images_array)



        # B, G, R = [sign_images_array[:, :, x] for x in range(3)]
        # [B, G, R] = [136, 167, 91]
        # non_mask_pix = sign_images_array != 0  # select everything that is not mask_value
        # mask_pix = sign_images_array == 0
        # sign_images_array[non_mask_pix] = [136, 167, 91] #genHumanSkinColor()
        # sign_images_array[mask_pix] = [111, 11, 4]   #genBackground()

        # cv2.imshow("Test-image", sign_images_array)
        # cv2.waitKey()