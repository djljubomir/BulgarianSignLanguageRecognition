import imutils
from PIL import ImageFont, ImageDraw, Image
import PILasOPENCV as Image
import PILasOPENCV as ImageDraw
import PILasOPENCV as ImageFont
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import numpy as np
import handy
import cv2
import pickle
import math
import time

# model = tf.keras.models.load_model('SLR-2E-128x128-Dropout-GrayScale_with_gen_background-CNN.model')
# model = tf.keras.models.load_model('SLR-1E-350x300-Dropout-CNN.model')

# model = pickle.load(open('SLR-128x128-Full-kNN_clf', 'rb')) #kNN
# model = pickle.load(open('SLR-128x128-Full_SVM_clf', 'rb'))  #SVM

# model = tf.keras.models.load_model('SLR-7E-CNN.model')
# sign_names = ["а", "б", "в", "г", "д", "е", "ж", "з", "и", "й", "к", "л", "м", "н", "о", "п", "р", "с", "т", "у", "ф",
#               "х", "ц", "ч", "ш", "щ", "ъ", "ь", "ю", "я", "интервал", "изтриване", "нищо"]
#
# sign_categories = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11", "L12", "L13", "L14", "L15",
#                    "L16", "L17", "L18", "L19", "L20", "L21", "L22", "L23", "L24", "L25", "L26", "L27", "L28", "L29",
#                    "L30", "Space", "Delete", "Nothing"]


model = tf.keras.models.load_model('SLR-3E-2ndConv64-128x128-Full-CNN.model')
sign_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "Интервал", "а", "б", "в", "г", "д", "е", "ж", "з", "и",
              "й", "к", "л", "м", "н", "о", "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ь", "ю", "я",
              "Нищо", "аз", "да", "добре", "защо", "или", "как", "какво", "кога", "кой", "кое",  "колко",
              "къде", "между", "не", "след", "това", "тогава", "този", "ти", "той/тя"]

#"кого/кои/кой/коя" "този/тази/тези" "той/тя"

sign_categories = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "Interval", "L1", "L2", "L3", "L4", "L5", "L6",
                   "L7", "L8", "L9", "L10", "L11", "L12", "L13", "L14", "L15", "L16", "L17", "L18", "L19", "L20", "L21",
                   "L22", "L23", "L24", "L25", "L26", "L27", "L28", "L29", "L30", "Nothing", "W1", "W2", "W3", "W4",
                   "W5", "W6", "W7", "W8", "W9", "W10", "W11", "W12", "W13", "W14", "W15", "W16", "W17", "W18", "W19",
                   "W20"]



def Rule_A(BGR_Frame):
    '''this function implements the RGB bounding rule algorithm
    --inputs:
    BGR_Frame: BGR components of an image
    plot: Bool type variable,if set to True draw the output of the algorithm
    --return a anumpy array of type bool like the following:
    [[False False False True]
    [False False False True]
    .
    .
    .
    [False False False True]]
    2d numpy array
    So in order to plot this matrix, we need to convert it to numbers like:
    255 for True values(white)
    0 for False(black)
    '''
    # B_Frame, G_Frame, R_Frame = [BGR_Frame[..., BGR] for BGR in range(3)]  # [...] is the same as [:,:]
    # # you can use the split built-in method in cv2 library to get the b,g,r components
    # # B_Frame, G_Frame, R_Frame  = cv2.split(BGR_Frame)
    # # i am using reduce built in method to get the maximum of a 3 given matrices
    # BRG_Max = np.maximum.reduce([B_Frame, G_Frame, R_Frame])
    # BRG_Min = np.minimum.reduce([B_Frame, G_Frame, R_Frame])
    # # at uniform daylight, The skin colour illumination's rule is defined by the following equation :
    # Rule_1 = np.logical_and.reduce([R_Frame > 95, G_Frame > 40, B_Frame > 20,
    #                                 BRG_Max - BRG_Min > 15, abs(R_Frame - G_Frame) > 15,
    #                                 R_Frame > G_Frame, R_Frame > B_Frame])
    # # the skin colour under flashlight or daylight lateral illumination rule is defined by the following equation :
    # Rule_2 = np.logical_and.reduce([R_Frame > 220, G_Frame > 210, B_Frame > 170,
    #                                 abs(R_Frame - G_Frame) <= 15, R_Frame > B_Frame, G_Frame > B_Frame])
    # # Rule_1 U Rule_2
    # RGB_Rule = np.logical_or(Rule_1, Rule_2)
    # return RGB_Rule


def vectorized_form(img):
    B, G, R = [img[:,:,x] for x in range(3)]
    delta15 = np.abs(R.astype(np.int8) - G.astype(np.int8)) > 15
    more_R_than_B = (R > B)
    is_skin_coloured_during_daytime = ((R > 95) & (G > 40) & (B > 20) &
        (img.ptp(axis=-1) > 15) & delta15 & (R > G) & more_R_than_B)
    is_skin_coloured_under_flashlight = ((R > 220) & (G > 210) & (B > 170) &
        ~delta15 & more_R_than_B & (G > B))
    return (is_skin_coloured_during_daytime | is_skin_coloured_under_flashlight).astype(int)


def lines(axis):
    '''return a list of lines for a give axis'''
    # equation(3)
    line1 = 1.5862 * axis + 20
    # equation(4)
    line2 = 0.3448 * axis + 76.2069
    # equation(5)
    # the slope of this equation is not correct Cr ≥ -4.5652 × Cb + 234.5652
    # it should be around -1
    line3 = -1.005 * axis + 234.5652
    # equation(6)
    line4 = -1.15 * axis + 301.75
    # equation(7)
    line5 = -2.2857 * axis + 432.85
    return [line1, line2, line3, line4, line5]

def YCrBr_RuleB(img):
    Y_Frame, Cr_Frame, Cb_Frame = [image[..., YCrCb] for YCrCb in range(3)]
    line1, line2, line3, line4, line5 = lines(Cb_Frame)
    return np.logical_and.reduce([line1 - Cr_Frame >= 0,
                                  line2 - Cr_Frame <= 0,
                                  line3 - Cr_Frame <= 0,
                                  line4 - Cr_Frame >= 0,
                                  line5 - Cr_Frame >= 0])


def HSV_RuleC(img):
    Hue, Sat, Val = [img[..., i] for i in range(3)]
    return np.logical_or(Hue < 50, Hue > 150)


def nothing(x):
    pass


# cv2.namedWindow('Camera Output', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Camera Output', 600, 600)
# TrackBars for fixing skin color of the person
# cv2.createTrackbar('B for min', 'Camera Output', 0, 255, nothing)
# cv2.createTrackbar('G for min', 'Camera Output', 0, 255, nothing)
# cv2.createTrackbar('R for min', 'Camera Output', 0, 255, nothing)
# cv2.createTrackbar('B for max', 'Camera Output', 0, 255, nothing)
# cv2.createTrackbar('G for max', 'Camera Output', 0, 255, nothing)
# cv2.createTrackbar('R for max', 'Camera Output', 0, 255, nothing)

# cv2.createTrackbar('H for min', 'Camera Output', 0, 255, nothing)
# cv2.createTrackbar('S for min', 'Camera Output', 0, 255, nothing)
# cv2.createTrackbar('V for min', 'Camera Output', 0, 255, nothing)
# cv2.createTrackbar('H for max', 'Camera Output', 0, 255, nothing)
# cv2.createTrackbar('S for max', 'Camera Output', 0, 255, nothing)
# cv2.createTrackbar('V for max', 'Camera Output', 0, 255, nothing)

# Default skin color values in natural lighting
# cv2.setTrackbarPos('B for min', 'Camera Output',52)
# cv2.setTrackbarPos('G for min', 'Camera Output',128)
# cv2.setTrackbarPos('R for min', 'Camera Output',0)
# cv2.setTrackbarPos('B for max', 'Camera Output',255)
# cv2.setTrackbarPos('G for max', 'Camera Output',140)
# cv2.setTrackbarPos('R for max', 'Camera Output',146)

# cv2.namedWindow('Camera Output')
# cv2.namedWindow('HandTrain')
# Default skin color values in indoor lighting
# cv2.setTrackbarPos('B for min', 'Camera Output', 0)
# cv2.setTrackbarPos('G for min', 'Camera Output', 133)
# cv2.setTrackbarPos('R for min', 'Camera Output', 77)
# cv2.setTrackbarPos('B for max', 'Camera Output', 255)
# cv2.setTrackbarPos('G for max', 'Camera Output', 167)
# cv2.setTrackbarPos('R for max', 'Camera Output', 127)

# cv2.setTrackbarPos('H for min', 'Camera Output', 0)
# cv2.setTrackbarPos('S for min', 'Camera Output', 48)
# cv2.setTrackbarPos('V for min', 'Camera Output', 80)
# cv2.setTrackbarPos('H for max', 'Camera Output', 20)
# cv2.setTrackbarPos('S for max', 'Camera Output', 255)
# cv2.setTrackbarPos('V for max', 'Camera Output', 255)

cap = cv2.VideoCapture(0)
#hist = handy.capture_histogram(source=0)

frame_index = 0
prediction = 0
predictions = []
all_predictions_arr = []
max_width = 1500
accept = False
space = False
delete = False
while True:
    # try:
    frame_index = frame_index + 1
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (1750, 850))

    cv2.rectangle(frame, (1500, 650), (795, 45), (124, 252, 0), 0)

    sign_width = 0
    row = 0
    if frame_index % 1 == 0: #and sign_names[prediction] != "Нищо":
        # for pred in all_predictions_arr:
        font = ImageFont.truetype("arial.ttf", 30)
        print(font)
        im = Image.new("RGB", (512, 512), "grey")
        draw = ImageDraw.Draw(im)

        # symbol = sign_names[prediction]
        # symbols_arr.append(symbol)
        #
        # text = pred
        # if sign_names[prediction] == "Интервал":
        #     sign_width += 66

        for sym in all_predictions_arr:
            draw.text((100, 250), sym, font=font, fill=(0, 0, 0))

            if sym != "Интервал":
                mask = ImageFont.getmask(sym, font)
            else:
                mask = (0, 0, 0)

            cv2.imwrite("sign" + '.jpeg', mask)
            img = cv2.imread('sign' + '.jpeg')
            img = cv2.resize(img, (64, 64))

            frame[700:764, 6 + sign_width:70 + sign_width] = img
            sign_width += 66
            if sign_width + 7 >= max_width:
                all_predictions_arr = []
                sign_width = 0


            # if sign_width + 7 < max_width:
            #     frame[675 + row*70:739 + row*70, 6+sign_width:70+sign_width] = img
            #     sign_width += 66
            # elif row == 1:
            #     all_predictions_arr = []
            #     sign_width = 0
            # else:
            #     row = 1
            #     frame[750:814, 6:70] = img
            #     sign_width = 66

    ##################################SENTENCE###########################################################
    #     if sign_names[prediction] != "изтриване":
    #         predictions.append(img)
    #     else:
    #         predictions.pop()
    #
        # for current_width in range(6, len(predictions)*65, 65):
        #     frame[760:824, current_width:current_width+64] = predictions[int((current_width-6) / 65)]
        #     max_width = current_width
        #     if current_width >= 1372:
        #         current_width = 6
        #         predictions = []
        #         max_width = 1500
##########################SENTENCE##################################################

    # hand = frame[50:650, 800:1500].copy()

    # height, width, channels = hand.shape

    # Accessing BGR pixel values
    # for x in range(0, 600):
    #     for y in range(0, 700):
    #         print(hand[x, y, 0])  # B Channel Value
    #         print(hand[x, y, 1])  # G Channel Value
    #         print(hand[x, y, 2])  # R Channel Value

    # bgModel = cv2.createBackgroundSubtractorMOG2(0, 50)
    # fgmask = bgModel.apply(frame[50:650, 800:1500])
    # kernel = np.ones((3, 3), np.uint8)
    # fgmask = cv2.erode(fgmask, kernel, iterations=4)
    # img = cv2.bitwise_and(frame[50:650, 800:1500], frame[50:650, 800:1500], mask=fgmask)
    #
    # hsv = cv2.cvtColor(img, cv2.cv2.COLOR_BGR2YCR_CB)
    # lower = np.array([20, 133, 77], dtype="uint8")
    # upper = np.array([200, 255, 255], dtype="uint8")
    # lower = np.array([cv2.getTrackbarPos('B for min', 'Camera Output'),
    #                          cv2.getTrackbarPos('G for min', 'Camera Output'),
    #                          cv2.getTrackbarPos('R for min', 'Camera Output')], np.uint8)
    #
    # upper = np.array([cv2.getTrackbarPos('B for max', 'Camera Output'),
    #                          cv2.getTrackbarPos('G for max', 'Camera Output'),
    #                          cv2.getTrackbarPos('R for max', 'Camera Output')], np.uint8)


    # hand = cv2.inRange(hsv, lower, upper)

    # min_YCrCb = numpy.array([cv2.getTrackbarPos('B for min', 'Camera Output'),
    #                          cv2.getTrackbarPos('G for min', 'Camera Output'),
    #                          cv2.getTrackbarPos('R for min', 'Camera Output')], numpy.uint8)
    # max_YCrCb = numpy.array([cv2.getTrackbarPos('B for max', 'Camera Output'),
    #                          cv2.getTrackbarPos('G for max', 'Camera Output'),
    #                          cv2.getTrackbarPos('R for max', 'Camera Output')], numpy.uint8)
    # extrapolate the hand to fill dark spots within
    # hand = cv2.dilate(hand, kernel, iterations=4)
    #
    # # blur the image
    # hand = cv2.GaussianBlur(hand, (5, 5), 100)
    # #
    # contours, hierarchy = cv2.findContours(hand, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    # skin_mask = vectorized_form(hand)
    # skin_mask = np.array(skin_mask, dtype="uint8")
    # hand = cv2.bitwise_and(frame[50:650, 800:1500], frame[50:650, 800:1500], mask=skin_mask)

    # contours, hierarchy = cv2.findContours(skin_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #
    # # find contour of max area(hand)
    # cnt = max(contours, key=lambda x: cv2.contourArea(x))
    #
    # # approx the contour a little
    # epsilon = 0.0005 * cv2.arcLength(cnt, True)
    # approx = cv2.approxPolyDP(cnt, epsilon, True)
    #
    # # make convex hull around hand
    # hull = cv2.convexHull(cnt)
    #
    # # define area of hull and area of hand
    # areahull = cv2.contourArea(hull)
    # areacnt = cv2.contourArea(cnt)
    #
    # # find the percentage of area not covered by hand in convex hull
    # arearatio = ((areahull - areacnt) / areacnt) * 100
    #
    # # find the defects in convex hull with respect to hand
    # hull = cv2.convexHull(approx, returnPoints=False)
    # defects = cv2.convexityDefects(approx, hull)
    #
    # # l = no. of defects
    # l = 0
    #
    # # code for finding no. of defects due to fingers
    # for i in range(defects.shape[0]):
    #     s, e, f, d = defects[i, 0]
    #     start = tuple(approx[s][0])
    #     end = tuple(approx[e][0])
    #     far = tuple(approx[f][0])
    #     pt = (100, 180)
    #
    #     # find length of all sides of triangle
    #     a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    #     b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
    #     c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
    #     s = (a + b + c) / 2
    #     ar = math.sqrt(s * (s - a) * (s - b) * (s - c))
    #
    #     # distance between point and convex hull
    #     d = (2 * ar) / a
    #
    #     # apply cosine rule here
    #     angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
    #
    #     # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
    #     if angle <= 90 and d > 30:
    #         l += 1
    #         cv2.circle(frame[50:650, 800:1500], far, 3, [255, 0, 0], -1)
    #
    #     # draw lines around hand
    #     cv2.line(frame[50:650, 800:1500], start, end, [0, 255, 0], 2)
    #
    # l += 1
    #
    # # print corresponding gestures which are in their ranges

    # image = frame[50:650, 800:1500].copy()
    image = frame[50:650, 800:1500]

    # Ycbcr_Frame = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # HSV_Frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # # Rule A ∩ Rule B ∩ Rule C
    # hand_arr = np.logical_and.reduce([vectorized_form(image), YCrBr_RuleB(Ycbcr_Frame), HSV_RuleC(HSV_Frame)])
    # non_mask_pix = hand_arr != 0  # select everything that is not mask_value
    # mask_pix = hand_arr == 0
    # image[non_mask_pix] = 255
    # image[mask_pix] = 0
    # cv2.imshow("Combined", image)

    # img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # hand_arr = HSV_RuleC(img_hsv)
    # dark_color = [1, 1, 1]
    # white = [255, 255, 255]
    #
    # non_mask_pix = hand_arr != 0  # select everything that is not mask_value
    # mask_pix = hand_arr == 0
    # image[non_mask_pix] = 255
    # image[mask_pix] = 0
    # cv2.imshow("HSV_image", image)

    # img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    # # blur = cv2.GaussianBlur(img_ycrcb, (3, 3), 0)
    # hand_arr = YCrBr_RuleB(img_ycrcb)
    #
    # non_mask_pix = hand_arr != 0  # select everything that is not mask_value
    # mask_pix = hand_arr == 0
    # image[non_mask_pix] = 255
    # image[mask_pix] = 0
    # cv2.imshow("YCrBr_image", image)



    hand_arr = vectorized_form(image)

    non_mask_pix = hand_arr != 0  # select everything that is not mask_value
    mask_pix = hand_arr == 0
    image[non_mask_pix] = 255
    image[mask_pix] = 0
    cv2.imshow("RGB", image)

    # print(vectorized_form(image))
    # skin = cv2.bitwise_and(image, image, mask=hand_arr)


    # stencil = np.zeros(hand_arr.shape[:-1]).astype(np.uint8)
    # non_mask_pix = stencil != 0  # select everything that is not mask_value
    # mask_pix = stencil == 0
    # hand_arr.reshape(-1, 600, 700, 3)
    # hand_arr[non_mask_pix] = dark_color
    # hand_arr[mask_pix] = white
    # for i in range(0, 600, 1):
    #     for j in range(0, 700, 1):
    #         if not hand_arr[i][j]:
    #             hnd[i][j] = dark_color
    #         else:
    #             hnd[i][j] = white



    # img_ycrcb = cv2.cvtColor(hand, cv2.COLOR_BGR2YCR_CB)
    # blur = cv2.GaussianBlur(img_ycrcb, (3, 3), 0)

    # img_ycrcb = cv2.cvtColor(hand, cv2.COLOR_BGR)

    # skin_ycrcb_min = np.array((0, 133, 77))
    # skin_ycrcb_max = np.array((255, 167, 133))
    #
    # skin_ycrcb_min = np.array((1, 1, 1))
    # skin_ycrcb_max = np.array((255, 255, 255))
    #
    # skin_ycrcb_min = np.array([cv2.getTrackbarPos('B for min', 'Camera Output'),
    #                          cv2.getTrackbarPos('G for min', 'Camera Output'),
    #                          cv2.getTrackbarPos('R for min', 'Camera Output')], np.uint8)
    #
    # skin_ycrcb_max = np.array([cv2.getTrackbarPos('B for max', 'Camera Output'),
    #                          cv2.getTrackbarPos('G for max', 'Camera Output'),
    #                          cv2.getTrackbarPos('R for max', 'Camera Output')], np.uint8)
    # hand = cv2.inRange(image, skin_ycrcb_min, skin_ycrcb_max)


    # print(hand)

    # lower = np.array([0, 48, 80], dtype="uint8")
    # upper = np.array([20, 255, 255], dtype="uint8")
    #
    # lower = np.array([cv2.getTrackbarPos('H for min', 'Camera Output'),
    #                          cv2.getTrackbarPos('S for min', 'Camera Output'),
    #                          cv2.getTrackbarPos('V for min', 'Camera Output')], np.uint8)
    #
    # upper = np.array([cv2.getTrackbarPos('H for max', 'Camera Output'),
    #                          cv2.getTrackbarPos('S for max', 'Camera Output'),
    #                          cv2.getTrackbarPos('V for max', 'Camera Output')], np.uint8)
    #
    # converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # hand = cv2.inRange(converted, lower, upper)

    # dark_color = [1, 1, 1]
    # white = [255, 255, 255]
    # stencil = np.zeros(hand.shape[:-1]).astype(np.uint8)
    # cv2.fillPoly(stencil, hand, white)
    # img = cv2.bitwise_and(hand, stencil, 0)
    #
    # non_mask_pix = stencil != 0  # select everything that is not mask_value
    # mask_pix = stencil == 0
    # img[non_mask_pix] = white
    # img[mask_pix] = dark_color

    # gray = cv2.cvtColor(frame[50:650, 800:1500], cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise




    # thresh = cv2.threshold(hand, 1, 255, cv2.THRESH_BINARY)[1]
    # # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    # thresh = cv2.erode(thresh, None, iterations=2)
    # thresh = cv2.dilate(thresh, None, iterations=2)
    # # # find contours in thresholded image, then grab the largest
    # # # one
    # contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = imutils.grab_contours(contours)
    # max_contour = 0
    # contour_area = 450000
    # if len(contours):
    #     max_contour = max(contours, key=cv2.contourArea)
    #     cv2.drawContours(frame[50:650, 800:1500], [max_contour], -1, (0, 255, 255), 2)
    #     dark_color = [1, 1, 1]
    #     white = [255, 255, 255]
    #     stencil = np.zeros(hand.shape[:-1]).astype(np.uint8)
    #
    #     non_mask_pix = stencil != 0  # select everything that is not mask_value
    #     mask_pix = stencil == 0
    #     hand[non_mask_pix] = 0
    #     hand[mask_pix] = 0
    #
    #     cv2.fillPoly(hand, [max_contour], white)
    #     # res = cv2.bitwise_and(image, stencil, 255)
    #     contour_area = cv2.contourArea(max_contour)





    # extLeft = tuple(c[c[:, :, 0].argmin()][0])
    # extRight = tuple(c[c[:, :, 0].argmax()][0])
    # extTop = tuple(c[c[:, :, 1].argmin()][0])
    # extBot = tuple(c[c[:, :, 1].argmax()][0])

    # draw the outline of the object, then draw each of the
    # extreme points, where the left-most is red, right-most
    # is green, top-most is blue, and bottom-most is teal

    # cv2.drawContours(image, [c], -1, (0, 255, 255), 2)

    # cv2.circle(image, extLeft, 8, (0, 0, 255), -1)
    # cv2.circle(image, extRight, 8, (0, 255, 0), -1)
    # cv2.circle(image, extTop, 8, (255, 0, 0), -1)
    # cv2.circle(image, extBot, 8, (255, 255, 0), -1)

    # show the output image
    # cv2.imshow("Hand", hand)

    # cv2.imshow("Image", image)

    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.imshow("images", hand)


# cv2.imshow("Hand", img)
    cv2.imshow("", frame)
    # except:
    #     pass
# if max_width % 1371 == 0:
#     cv2.waitKey(-1)
# cv2.imshow("", img)

    key = cv2.waitKey(1)
    if key == 27:
        break
    if key == ord('p'):
        cv2.waitKey(-1)
    if key == ord('a') and accept is False:
        accept = True
    if key == ord('c'):
        all_predictions_arr = []
        accept = False
    if key == ord('d') and all_predictions_arr:
        all_predictions_arr.pop()
        accept = False
        delete = True
    if key == ord('s') and sign_width + 67 <= max_width:
        if accept is False and all_predictions_arr and delete is False:
            all_predictions_arr.pop()
            all_predictions_arr.append(sign_names[10])
            if sign_names[prediction] != "Нищо":
                all_predictions_arr.append(sign_names[prediction])
        else:
            all_predictions_arr.append(sign_names[10])
        space = True


    # elif key == ord('a'):

    # if frame_index % 33 == 0 and contour_area < 400000:
    #     cv2.imwrite(str(frame_index) + '.jpeg', frame)
    #     img = cv2.imread(str(frame_index) + '.jpeg')
    #     crop_img = img[53:647, 803:1497]
    #     cv2.imwrite(str(frame_index) + 'cropped' + '.jpeg', crop_img)
    #
    #     test_img = cv2.imread(str(frame_index) + 'cropped' + '.jpeg',  cv2.IMREAD_GRAYSCALE)
    #     test_img = cv2.resize(test_img, (128, 128))
    #     test_img = test_img / 255
    #
    #     # test_data = []
    #     # test_data.append(test_img)
    #     predictions_array = model.predict([test_img.reshape(-1, 128, 128, 1)])
    #     print(predictions_array)
    #
    #     prediction = np.argmax(predictions_array[0])
    #     print(sign_names[prediction])
    #     print(contour_area)

    if frame_index % 20 == 0: #and contour_area < 420000:
        cv2.imwrite(str(frame_index)+ '.jpeg', image)
        test_img = cv2.imread(str(frame_index) + '.jpeg', cv2.IMREAD_GRAYSCALE)
        # test_img = cv2.flip(test_img, 1)

        test_img = cv2.resize(test_img, (128, 128))
        test_img = test_img / 255
        # test_img = cv2.resize(test_img, (1, 16384))

        test_data = []
        test_data.append(test_img)

        predictions_array = model.predict([test_img.reshape(-1, 128, 128, 1)]) #CNN
        print(predictions_array)

        prediction = np.argmax(predictions_array[0])

        if accept is False and all_predictions_arr and delete is False:
            all_predictions_arr.pop()

        if space is True:
            space = False

        if accept is True:
            accept = False

        if delete is True:
            delete = False

        if sign_names[prediction] != "Нищо":
            all_predictions_arr.append(sign_names[prediction])

        # if prediction > 40:
        #     all_predictions_arr.append(sign_names[prediction])

        # print(sign_names[prediction])
        print(all_predictions_arr)

#####

        # predictions_arr = model.predict(test_img.reshape(1, 16384)) #kNN and SVM
        # prediction = predictions_arr[0]
        # all_predictions_arr.append(sign_names[prediction])
        #
        # print(sign_names[prediction])
        # print(all_predictions_arr)


        # print(contour_area)
        # cv2.imshow("", test_img)

    #cv2.imshow("", crop_img)

    #     cv2.destroyAllWindows()
    #     cap.release()
    #     break

# img_ycrcb = cv2.cvtColor(hand, cv2.COLOR_BGR2YCR_CB)
# 	blur = cv2.GaussianBlur(img_ycrcb,(11,11),0)
# 	skin_ycrcb_min = np.array((0, 138, 67))
# 	skin_ycrcb_max = np.array((255, 173, 133))
# 	mask = cv2.inRange(blur, skin_ycrcb_min, skin_ycrcb_max)  # detecting the hand in the bounding box using skin detection
	# contours,hierarchy = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL, 2)
# 	cnt=ut.getMaxContour(contours,4000)
