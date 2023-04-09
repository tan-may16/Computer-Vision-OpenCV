import cv2
import numpy as np
import matplotlib.pyplot as plt

################################################################################

#Motion tracking using contours on live feed
def blank(x):
    pass
cap=cv2.VideoCapture(0)

_,frame1=cap.read()
_,frame2=cap.read()
cv2.namedWindow('Trackbars')
cv2.createTrackbar('kernal','Trackbars',0,10,blank)
cv2.createTrackbar('Sigma','Trackbars',0,10,blank)
cv2.createTrackbar('threshold','Trackbars',0,255,blank)

while(cap.isOpened()):
    k=cv2.getTrackbarPos('kernal','Trackbars')
    sigma=cv2.getTrackbarPos('Sigma','Trackbars')
    T=cv2.getTrackbarPos('threshold','Trackbars')
    frame_diff=cv2.absdiff(frame1,frame2)
    diff_gray=cv2.cvtColor(frame_diff,cv2.COLOR_BGR2GRAY)
    gaussian_blur=cv2.GaussianBlur(diff_gray,(2*k +1,2*k +1),sigma)
    _,threshold=cv2.threshold(gaussian_blur,T,255,cv2.THRESH_BINARY)
    dilated_thresh=cv2.dilate(threshold,None,iterations=3)
    contours,_=cv2.findContours(dilated_thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame1,contours,-1,(255,0,0),1)
    cv2.imshow('Live tracking',frame1)
    frame1=frame2
    _,frame2=cap.read()

    k=cv2.waitKey(10)
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()

################################################################################

###Shape detection
img=cv2.imread('shapes.jfif')
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
def blank(x):
    pass
cv2.namedWindow('shape detection')
cv2.createTrackbar('threshold','shape detection',0,255,blank)
while True:
    t=cv2.getTrackbarPos('threshold','shape detection')
    _,threshold=cv2.threshold(img_gray,t,255,cv2.THRESH_BINARY)
    contours,_=cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for i in contours:
        approx_poly=cv2.approxPolyDP(i,0.01*cv2.arcLength(i,True),True)
        cv2.drawContours(img,[approx_poly],0,(255,0,0),1)
    cv2.imshow('shape detection',img)
    k=cv2.waitKey(0)
    if k==27:
        break
cv2.destroyAllWindows()

################################################################################

###Histograms
img=cv2.imread('photo2.png',0)
# plt.hist(img.ravel(),256,[0,256])
histogram=cv2.calcHist([img],[0],None,[256],[0,256])
plt.plot(histogram)
plt.show()

################################################################################

###Template Matching
def blank(x):
    pass
img=cv2.imread('photo2.png',0)
img_original=cv2.imread('photo2.png')
template=cv2.imread('template.png',0)
matching=cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
cv2.namedWindow('Template_Matching')
cv2.createTrackbar('threshold','Template_Matching',90,100,blank)
cv2.createTrackbar('switch','Template_Matching',0,1,blank)
while True:
    t=cv2.getTrackbarPos('threshold','Template_Matching')
    switch=cv2.getTrackbarPos('switch','Template_Matching')
    filtered_match=np.where(matching>=t/100)
    cv2.imshow('Template_Matching',img_original)
    for i in zip(*filtered_match):
        cv2.rectangle(img_original,i,(i[0]+template.shape[0],i[1]+template.shape[1]),(0,0,255),1)
    if switch==1:
        img_original=cv2.imread('photo2.png')
    k=cv2.waitKey(1)
    if k==27:
        break
cv2.destroyAllWindows()

################################################################################


###Hough Line Transform
def blank(x):
    pass
img_original_1=cv2.imread('chessboard.jpg')
img_original_2=cv2.imread('chessboard.jpg')
img=cv2.imread('chessboard.jpg',0)
cv2.namedWindow('hough line transform')
cv2.createTrackbar('L','hough line transform',50,255,blank)
cv2.createTrackbar('H','hough line transform',130,255,blank)
cv2.createTrackbar('switch','hough line transform',0,1,blank)
while True:
    l=cv2.getTrackbarPos('L','hough line transform')
    h=cv2.getTrackbarPos('H','hough line transform')
    switch=cv2.getTrackbarPos('switch','hough line transform')
    canny_edges=cv2.Canny(img,l,h,apertureSize=3)
    h_lines=cv2.HoughLines(canny_edges,1,np.pi/180,200)
    hp_lines=cv2.HoughLinesP(canny_edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)


    for i in h_lines:
        r,theta=i[0]
        xl=r*np.cos(theta)  ##Top left corner of image
        yl=r*np.sin(theta)

        x1=int(xl + 1000*(-r*np.sin(theta)))
        y1=int(yl + 1000*(r*np.cos(theta)))
        x2=int(xl - 1000*(-r*np.sin(theta)))
        y2=int(yl - 1000*(r*np.cos(theta)))
        cv2.line(img_original_1,(x1,y1),(x2,y2),(0,255,255),1)
    
    for j in hp_lines:
        x1,y1,x2,y2=j[0]
        cv2.line(img_original_2,(x1,y1),(x2,y2),(0,255,255),1)
    
    if switch==1:
        img_original=cv2.imread('chessboard.jpg')
    cv2.imshow('hough line transform',img_original_1)
    cv2.imshow('hough prob line transform',img_original_2)
    k=cv2.waitKey(1)
    if k==27:
        break
cv2.destroyAllWindows()

################################################################################

###Lane detection

cap=cv2.VideoCapture('road.mp4')
_,frame1=cap.read()

def roi(img,vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask,vertices,(255,))
    roi=cv2.bitwise_and(img,mask)
    return roi
h,w,channels=frame1.shape
vertices=[(0,h),(0,3*h/4),(2.5*w/6,h/4),(w,h)]
roi_frame1=roi(frame1,np.array([vertices],np.int32))

while(cap.isOpened()):
    _,frame=cap.read()
    frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame_canny=cv2.Canny(frame_gray,100,150)
    roi_frame=roi(frame_canny,np.array([vertices],np.int32))
    hp_lines=cv2.HoughLinesP(roi_frame,1,np.pi/180,200,np.array([]),minLineLength=40,maxLineGap=25)
    for i in hp_lines:
        x1,y1,x2,y2=i[0]
        cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),1)
    cv2.imshow('Canny edges',roi_frame)
    cv2.imshow('Lane Detection',frame)
    k=cv2.waitKey(50)
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()

################################################################################


###Face_detection using cascade classifier

face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)
while cap.isOpened():
    _,frame=cap.read()
    frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(frame_gray,1.1,4)

    for i in faces:
        x,y,w,h=i
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imshow('Face detection',frame)
    k=cv2.waitKey(1)
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()


################################################################################

###Corner Detection

##Haris Corner
img=cv2.imread('chessboard.jpg')
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_corner=cv2.cornerHarris(np.float32(img_gray),2,5,0.1)
dilate=cv2.dilate(img_corner,None)
img[dilate>0.01*dilate.max()]=[0,0,255]

##Shi Tomasi
shi_corners=cv2.goodFeaturesToTrack(img_gray,40,0.01,10)
shi_corners=np.int64(shi_corners)
for i in shi_corners:
    x,y=i.ravel()
    cv2.circle(img,(x,y),6,(255,0,0),-1)


cv2.imshow('Corner detection',img)
cv2.waitKey(0)

################################################################################


###Background Subtractor
cap=cv2.VideoCapture('road.mp4')
bgs=cv2.createBackgroundSubtractorMOG2()
while(cap.isOpened()):
    _,frame=cap.read()
    background_subtract=bgs.apply(frame)
    cv2.imshow('Background subtracton',background_subtract)
    k=cv2.waitKey(10)
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()


################################################################################
