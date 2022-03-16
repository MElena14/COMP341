import numpy as np
from numpy import cos, sin

import cv2
try:
    from google.colab.patches import cv2_imshow
    IN_COLAB = True
except:
    cv2_imshow = cv2.imshow
    IN_COLAB = False




def draw_box(image,center_point, rotate, width, height, color):


    angle = np.radians(rotate)
    # Determine the coordinates of the 4 corner points
    rotated_rect_points = []
    x = center_point[0] + ((width / 2) * cos(angle)) - ((height / 2) * sin(angle))
    y = center_point[1] + ((width / 2) * sin(angle)) + ((height / 2) * cos(angle))
    rotated_rect_points.append([x,y])
    x = center_point[0] - ((width / 2) * cos(angle)) - ((height / 2) * sin(angle))
    y = center_point[1] - ((width / 2) * sin(angle)) + ((height / 2) * cos(angle))
    rotated_rect_points.append([x,y])
    x = center_point[0] - ((width / 2) * cos(angle)) + ((height / 2) * sin(angle))
    y = center_point[1]- ((width / 2) * sin(angle)) - ((height / 2) * cos(angle))
    rotated_rect_points.append([x,y])
    x = center_point[0] + ((width / 2) * cos(angle)) + ((height / 2) * sin(angle))
    y = center_point[1] + ((width / 2) * sin(angle)) - ((height / 2) * cos(angle))
    rotated_rect_points.append([x,y])

    rotatedImg = cv2.polylines(image, np.array([rotated_rect_points], np.int32), True, color, thickness)
    return rotatedImg


def rotated_rectangle(image, predicted, correct, thickness,):

    predictedColor = (0, 255 ,0)
    predicted_center_point = predicted[0],predicted[1]
    predicted_rotate = predicted[2]
    predicted_width = predicted[3]
    predicted_height = predicted[4]

    predicted_result = draw_box(image, predicted_center_point, predicted_rotate, predicted_width, predicted_height, predictedColor)

    correctColor = (0, 0 ,255)
    correct_center_point = correct[0],correct[1]
    correct_rotate = correct[2]
    correct_width = correct[3]
    correct_height = correct[4]

    final = draw_box(predicted_result, correct_center_point, correct_rotate, correct_width, correct_height, correctColor)

    cv2_imshow(final)






# Prediction array [Χ, Υ, angle, width, height, color ]

correct = np.array([671.37889, 747.50826, 66.6416, 18.0, 81.4823])
prediction = np.array([470.14815, 253.89551, 68.2615, 25.5, 25.6211])

img = cv2.imread('/content/0_1a9fa4c269cfcc1b738e43095496b061_RGB.png')




rotated_rectangle(img, prediction, correct, 2)

# Second try
center_point2 = 671.37889,747.50826
angle2 = 66.6416
width2 = 18.0
height2 = 81.4823
color = list(np.random.random(size=3) * 256)
thickness = 2

#rotated_rectangle(img, center_point2, height2, width2, color, thickness, angle2)

# Third Try

center_point3 = 621.03369,479.57938
angle3 = -23.38
width3 = 19.5
height3 = 12.9206
color = list(np.random.random(size=3) * 256)
thickness = 2

#rotated_rectangle(img, center_point3, height3, width3, color, thickness, angle3)
