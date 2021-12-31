import numpy as np
import cv2
import matplotlib.pyplot as plt
################################################################################
### Color tracking using HSV colorspace
def blank(x):
    pass

cv2.namedWindow('Masking')
cv2.createTrackbar('LH','Masking',0,255,blank)
cv2.createTrackbar('HH','Masking',255,255,blank)

cv2.createTrackbar('LS','Masking',0,255,blank)
cv2.createTrackbar('HS','Masking',255,255,blank)

cv2.createTrackbar('LV','Masking',0,255,blank)
cv2.createTrackbar('HV','Masking',255,255,blank)
##If color range to be tracked is already known, no need of trackbars
while True:
    img=cv2.imread('photo2.png')
    img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    lh=cv2.getTrackbarPos('LH','Masking')
    hh=cv2.getTrackbarPos('HH','Masking')
    ls=cv2.getTrackbarPos('LS','Masking')
    hs=cv2.getTrackbarPos('HS','Masking')
    lv=cv2.getTrackbarPos('LV','Masking')
    hv=cv2.getTrackbarPos('HV','Masking')

    LowerBound=np.array([lh,ls,lv])
    UpperBound=np.array([hh,hs,hv])
    mask=cv2.inRange(img_hsv,LowerBound,UpperBound)
    img_masked=cv2.bitwise_and(img_hsv,img_hsv,mask=mask)
    cv2.imshow("Image",img_masked)
    cv2.imshow("Mask",mask)
    k=cv2.waitKey(1)
    if k==27:
        break
cv2.destroyAllWindows()

################################################################################

###Global Thresholding
img=cv2.imread('photo2.png')
_,img_thresh1=cv2.threshold(img,127,255,cv2.THRESH_BINARY) ##Converts image into pure RGB | Either 1 or 0
_,img_thresh2=cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV) ##Inverse of Binary
_,img_thresh3=cv2.threshold(img,127,255,cv2.THRESH_TRUNC) ##No change before threshold
_,img_thresh4=cv2.threshold(img,127,255,cv2.THRESH_TOZERO) ###Less than threshold value to zero
_,img_thresh5=cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV) ###Opposite of ToZero
cv2.imshow('Original image',img)
cv2.imshow('Binary Threshold',img_thresh1)
cv2.imshow('Binary Threshold Inverse',img_thresh2)
cv2.imshow('Binary Threshold Trunc',img_thresh3)
cv2.imshow('Binary Threshold To Zero',img_thresh4)
cv2.imshow('Binary Threshold To Zero inverse',img_thresh5)
k=cv2.waitKey(0)
if k==27:
    cv2.destroyAllWindows()

################################################################################

 ###Adaptive/Local Thresholding
img=cv2.imread('photo2.png',0)

img_thresh1=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,0) 
img_thresh2=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,9,0) 
cv2.imshow('Binary Threshold with adaptive mean',img_thresh1)
cv2.imshow('Binary Threshold with adaptive gaussian',img_thresh2)

k=cv2.waitKey(0)
if k==27:
    cv2.destroyAllWindows()

 ################################################################################

###Morphological Transformations

##Dilation and Erosion
img=cv2.imread('photo2.png',0)
_,thresh=cv2.threshold(img,150,255,cv2.THRESH_BINARY)
kernal=np.ones((4,4),np.uint8)
dilated_img=cv2.dilate(thresh,kernal,iterations=2)   ###Convolution by kernal
eroded_img=cv2.erode(thresh,kernal,iterations=1)  ###Convolution such that if all neighbeoring pixels have same value, original value retained
##MorphologyEx is general function.
open_img=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernal)  ##Erosion+dilation
close_img=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernal) ##Dilation+Erosion
gradient_img=cv2.morphologyEx(thresh,cv2.MORPH_GRADIENT,kernal) ###Dilation-Erosion
cv2.imshow('Original Image',img)
cv2.imshow('Mask',thresh)
cv2.imshow('Dilated Image',dilated_img)
cv2.imshow('Eroded Image',eroded_img) ##Boundaries are smoothened
cv2.imshow('Opening Image',open_img)  
cv2.imshow('Closing Image',close_img)  
cv2.imshow('Gradient Image',gradient_img) 
k=cv2.waitKey(0)
if k==27:
    cv2.destroyAllWindows()

################################################################################

###Image Smoothing

img=cv2.imread('photo1.png')
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
kernal=np.ones((4,4),np.float32)/16
filter2D=cv2.filter2D(img,-1,kernal)   ##Homogeneous Kernal
averaging=cv2.blur(img,(4,4))  ###Averaging
gaussian_blur=cv2.GaussianBlur(img,(5,5),0)   ##Gaussian Filter
median_filter=cv2.medianBlur(img,5)  ##Median Filter
bilateral_filter=cv2.bilateralFilter(img,5,100,100)

image_names=['original','2D convolution','Averaging','G aussian Blur','Median Filter','biltaeral Filter']
images=[img,filter2D,averaging,gaussian_blur,median_filter,bilateral_filter]
for i in range(len(image_names)):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i])
    plt.title(image_names[i])
    plt.xticks([])
    plt.yticks([])
plt.show()

# cv2.imshow('Original Image',img)
# cv2.imshow('2D convolution Image',final_img)
# cv2.imshow('Averaging of Image',averaging)
# k=cv2.waitKey(0)
# if k==27:
#     cv2.destroyAllWindows()

################################################################################

###Image Gradients

img=cv2.imread('photo1.png',0)
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
laplacian_gradient_temp=cv2.Laplacian(img,cv2.CV_64F,1)  ##Laplacian
laplacian_gradient=np.uint8(np.abs(laplacian_gradient_temp))
X_sobel=cv2.Sobel(img,cv2.CV_64F,1,0)   ##Sobel X
Sobel_X=np.uint8(np.abs(X_sobel))
Y_sobel=cv2.Sobel(img,cv2.CV_64F,0,1)   ##Sobel Y
Sobel_Y=np.uint8(np.abs(Y_sobel))

image_names=['Original','Laplacian gradient','Sobel X','Sobel Y']
images=[img,laplacian_gradient,Sobel_X,Sobel_X]
for i in range(len(image_names)):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i])
    plt.title(image_names[i])
    plt.xticks([])
    plt.yticks([])
plt.show()

################################################################################

###Canny Edge Detector

img=cv2.imread('photo1.png')
# img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
def blank(x):
    pass

cv2.namedWindow('Trackbars')
cv2.createTrackbar('L','Trackbars',0,255,blank)
cv2.createTrackbar('H','Trackbars',255,255,blank)


while True:
    l=cv2.getTrackbarPos('L','Trackbars')
    h=cv2.getTrackbarPos('H','Trackbars')
    canny_image=cv2.Canny(img,l,h)
    cv2.imshow('Original',img)
    cv2.imshow('Canny Edge',canny_image)
    k=cv2.waitKey(1)
    if k==27:
        break
cv2.destroyAllWindows()
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
image_names=['Original','Edge Detection']
images=[img,canny_image]
for i in range(len(image_names)):
    plt.subplot(2,1,i+1)
    plt.imshow(images[i],cmap='gray')
    plt.title(image_names[i])
    plt.xticks([])
    plt.yticks([])
plt.show()

################################################################################

###Image Pyramids- Creating images of different resolutions
img=cv2.imread('photo1.png')

#gaussian subsampling 
lower_res=cv2.pyrDown(img)
higher_res=cv2.pyrUp(img)

##Laplacian pyramids
res=cv2.pyrUp(lower_res)
laplacian_pyr=cv2.subtract(img,res)
cv2.imshow('original Image',img)
cv2.imshow('Lower resolution Image',lower_res)
cv2.imshow('Higher resolution Image',higher_res)
cv2.imshow('laplacian Image',laplacian_pyr)
k=cv2.waitKey(0)
if k==27:
    cv2.destroyAllWindows()

################################################################################

###Drawing Contours

##Everytime threshold is changed, change switch from 0 to 1 and back to 0
img=cv2.imread('photo2.png')
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
def blank(x):
    pass
cv2.namedWindow('Contour Image')
cv2.createTrackbar('Threshold','Contour Image',0,255,blank)
cv2.createTrackbar('Switch','Contour Image',0,1,blank)
while (True):
    T=cv2.getTrackbarPos('Threshold','Contour Image')
    S=cv2.getTrackbarPos('Switch','Contour Image')
    _,threshold=cv2.threshold(img_gray,T,255,0)
    contours,_=cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img,contours,-1,(255,0,0),1)
    if S==1:
        img=cv2.imread('photo2.png')
    cv2.imshow('Contour Image',img)
    k=cv2.waitKey(1)
    if k==27:
        break
cv2.destroyAllWindows()

################################################################################

