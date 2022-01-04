import cv2
import numpy as np

def gamma_correction(img, gamma):
    gamma_inverse = 1.0 / gamma  
    range=  np.arange(0, 256)
    lut = np.array([(((i / 255.0) ** gamma_inverse) * 255) for i in range]).astype("uint8")
    output=cv2.LUT(img, lut)
    return output
img_input = input('Enter the input image name:')    
img = cv2.imread(img_input)        
gamma = float(input('Enter gamma correction value: '))
img_gamma_corrected = gamma_correction(img, gamma ) 

cv2.imshow('Original Image',img)
cv2.imshow('Gamma Corrected',img_gamma_corrected)
cv2.waitKey(0)
cv2.destroyAllWindows()