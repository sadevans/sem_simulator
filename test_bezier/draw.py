import numpy as np

import cv2
import matplotlib.pyplot as plt


def simulate_circles():
    image_hole = np.zeros((120, 800), dtype=np.float32)
    center_coordinates1 = (40, 60)
    center_coordinates2 = (110, 60)
    center_coordinates3 = (200, 60)
    center_coordinates4 = (300, 60)
    center_coordinates5 = (400, 60)
    center_coordinates6 = (500, 60)
    center_coordinates7 = (600, 60)
    center_coordinates8 = (700, 60)



    radius = 20

    color_border = (128, 0, 0)
    color_hole = (255, 0, 0)
    
    thickness_border = -1
    thickness_hole = -1

    # border 3
    image_border = cv2.circle(image_hole, center_coordinates1, radius+3, color_border, 2)
    image_border = cv2.circle(image_border, center_coordinates1, radius+3, color_border, thickness_border)
    image_hole = cv2.circle(image_border, center_coordinates1, radius, color_hole,2)
    image_hole = cv2.circle(image_hole, center_coordinates1, radius, color_hole, thickness_hole)

    # border 4
    image_border = cv2.circle(image_hole, center_coordinates2, radius+4, color_border, 2)
    image_border = cv2.circle(image_border, center_coordinates2, radius+4, color_border, thickness_border)
    image_hole = cv2.circle(image_border, center_coordinates2, radius, color_hole,2)
    image_hole = cv2.circle(image_hole, center_coordinates2, radius, color_hole, thickness_hole)

    # border 5
    image_border = cv2.circle(image_hole, center_coordinates3, radius+5, color_border, 2)
    image_border = cv2.circle(image_border, center_coordinates3, radius+5, color_border, thickness_border)
    image_hole = cv2.circle(image_border, center_coordinates3, radius, color_hole,2)
    image_hole = cv2.circle(image_hole, center_coordinates3, radius, color_hole, thickness_hole)

    # border 6
    image_border = cv2.circle(image_hole, center_coordinates4, radius+6, color_border, 2)
    image_border = cv2.circle(image_border, center_coordinates4, radius+6, color_border, thickness_border)
    image_hole = cv2.circle(image_border, center_coordinates4, radius, color_hole,2)
    image_hole = cv2.circle(image_hole, center_coordinates4, radius, color_hole, thickness_hole)

    # border 7
    image_border = cv2.circle(image_hole, center_coordinates5, radius+7, color_border, 2)
    image_border = cv2.circle(image_border, center_coordinates5, radius+7, color_border, thickness_border)
    image_hole = cv2.circle(image_border, center_coordinates5, radius, color_hole,2)
    image_hole = cv2.circle(image_hole, center_coordinates5, radius, color_hole, thickness_hole)

    # border 8
    image_border = cv2.circle(image_hole, center_coordinates6, radius+8, color_border, 2)
    image_border = cv2.circle(image_border, center_coordinates6, radius+8, color_border, thickness_border)
    image_hole = cv2.circle(image_border, center_coordinates6, radius, color_hole,2)
    image_hole = cv2.circle(image_hole, center_coordinates6, radius, color_hole, thickness_hole)

    # border 9
    image_border = cv2.circle(image_hole, center_coordinates7, radius+9, color_border, 2)
    image_border = cv2.circle(image_border, center_coordinates7, radius+9, color_border, thickness_border)
    image_hole = cv2.circle(image_border, center_coordinates7, radius, color_hole,2)
    image_hole = cv2.circle(image_hole, center_coordinates7, radius, color_hole, thickness_hole)

    # border 10
    image_border = cv2.circle(image_hole, center_coordinates8, radius+10, color_border, 2)
    image_border = cv2.circle(image_border, center_coordinates8, radius+10, color_border, thickness_border)
    image_hole = cv2.circle(image_border, center_coordinates8, radius, color_hole,2)
    image_hole = cv2.circle(image_hole, center_coordinates8, radius, color_hole, thickness_hole)

    plt.imshow(image_hole, cmap='gray')
    plt.show()

    # cv2.imwrite('./data/test_img_two.png', image_border)
    return image_border

img_circles = simulate_circles()
cv2.imwrite('difr_circles.png', np.clip(img_circles, 0, 255))
