import cv2
import numpy as np


image_path = "./screenshots/341_rgb_maximum_of_even_odd.png"
image = cv2.imread(image_path)
b, g, r = cv2.split(image)

cv2.imwrite('blue.png', b)
cv2.imwrite('green.png', g)
cv2.imwrite('red.png', r)