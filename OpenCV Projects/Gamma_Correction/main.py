"""
@author: Tanmay Chinchanikar
"""


import cv2
import numpy as np

image_file = input('Enter the file name of the image: ')

input_image = cv2.imread(str(image_file))
cv2.imshow('Input Image', cv2.resize(input_image, None, fx=0.75, fy=0.75))


gamma = float(input('Enter a gamma value: '))

# Create the gamma table
inv_gamma = 1.0 / gamma
gamma_table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")


# Apply the gamma correction
gamma_corrected_image = cv2.LUT(input_image, gamma_table)
cv2.imshow('Gamma Corrected Image', cv2.resize(gamma_corrected_image, None, fx=0.6, fy=0.6))
cv2.waitKey(0)
cv2.destroyAllWindows()
