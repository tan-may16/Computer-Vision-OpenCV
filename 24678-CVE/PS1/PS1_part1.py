import cv2
import numpy as np
def blank(x):
    pass
img_name=input("Please Enter input Image Name:")
region=int(input("0:Dark Region  1:Bright Region"))
img=cv2.imread(img_name)
img_grayscale=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.namedWindow('Grayscale Image')
cv2.createTrackbar('Threshold','Grayscale Image',0,255,blank)
while True:
    t=cv2.getTrackbarPos('Threshold','Grayscale Image')
    _,thresh=cv2.threshold(img_grayscale,t,255,cv2.THRESH_BINARY)
    thresh_rgb=cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
    if region==1:
        thresh_rgb[np.where(thresh==255)]=[0,0,255]
        thresh_rgb[np.where(thresh==0)]=img[np.where(thresh==0)]
        cv2.imshow('Bright Image',thresh_rgb)
    elif region==0:
        thresh_rgb[np.where(thresh==0)]=[0,0,255]
        thresh_rgb[np.where(thresh==1)]=img[np.where(thresh==1)]
        cv2.imshow('Bright Image',thresh_rgb)
    cv2.imshow('Threshold Image',thresh)
    cv2.imshow('Grayscale Image',img_grayscale)
    cv2.imshow('Original Image',img)
    k=cv2.waitKey(1)
    if k==27:
        break
cv2.destroyAllWindows()
