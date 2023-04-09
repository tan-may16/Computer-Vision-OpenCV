import argparse
import os.path
import numpy as np
import cv2

parser = argparse.ArgumentParser(description='Convert an image to pseudo colored image')
parser.add_argument('image_path', type=str, help='path to the input image file')
args = parser.parse_args()

input_image = cv2.imread(args.image_path)
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
name, _ = os.path.splitext(os.path.basename(args.image_path))

grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
hsv_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2HSV)

h_lut = np.uint8(140*((grayscale_image-np.min(grayscale_image))/(np.max(grayscale_image)-np.min(grayscale_image))))

output_hsv_image = np.stack((h_lut, 255*np.ones_like(grayscale_image), 255*np.ones_like(grayscale_image)), axis=-1)

output_rgb_image = cv2.cvtColor(output_hsv_image, cv2.COLOR_HSV2RGB)

max_indices = np.argwhere(grayscale_image == np.max(grayscale_image))
max_index = max_indices[int(len(max_indices)/2)]

radius = 15
color = 255
thickness = 2
center = tuple(max_index[::-1])
marker_size = 80
grayscale_image = cv2.circle(grayscale_image, center, radius, color, thickness) 
grayscale_image = cv2.drawMarker(grayscale_image, center, color, 0, marker_size, thickness = 1 ) 

cv2.imshow('Original Image', input_image) 
cv2.imshow('Marked Image', grayscale_image) 
cv2.imshow('Pseudo Colored', output_rgb_image)
cv2.imwrite('pseudo_Colored_' + name + '.png', output_rgb_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
