import cv2
import numpy as np
import datetime 
################################################################################
##OpenCv version
print(cv2.__version__)

################################################################################

##Read and Show Images
img=cv2.imread("photo1.png") ###if 2nd argument is given 0, image in gray
cv2.imshow('Bird',img)

## Waitkey defined---  Close window on pressing ESC or s
k=cv2.waitKey(0)
if k==27:                           ###27=ESC key
    cv2.destroyAllWindows()
elif k==ord('s'):
    cv2.destroyAllWindows()

### Save image
cv2.imwrite('Bird_Copy.png',img)

################################################################################

###changing color space
img1=cv2.imread('photo1.png')
gray_bird=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray image',gray_bird)
k=cv2.waitKey(0)
if k==27:
    cv2.destroyAllWindows()

################################################################################

##Video Capture
cap=cv2.VideoCapture(0)                 ## Instead of 0, put video name. eg- video.mp4
fourcc=cv2.VideoWriter_fourcc(*'XVID')   ##saving video step 1
output=cv2.VideoWriter('Live Feed recording.avi',fourcc,25.0,(640,480))   ##saving video step 2
while(cap.isOpened()):                 ###instead of cap.isOpened(), True can be used
    _,frame=cap.read()
    if _==True:
        cv2.imshow("Live Feed",frame)
        # gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)     ###Gray video
        # cv2.imshow("Live Feed",gray_frame)
        output.write(frame)                 ##saving video step 3
        k=cv2.waitKey(1)
        if k==ord('s'):
            break
    else:
        break 
cap.release()
output.release()                    
cv2.destroyAllWindows()

################################################################################

##Geometry Shapes
# img=cv2.imread("photo1.png")
img=np.zeros((500,500,3),np.uint8) ##creating images with numpy
img_line=cv2.line(img,(0,0),(100,200),(255,255,255),2) ###Draw Line
img_line_Arrowed=cv2.arrowedLine(img,(0,0),(100,100),(0,255,255),2) ###Draw Line point to second coordinate
img_rectangle=cv2.rectangle(img, (50,50),(300,300),(0,255,0),3)  ### instead of 3, if thickness=-1, filled rectangle
img_circle=cv2.circle(img,(200,200),30,(255,255,255),-1)  ###Circle
font=cv2.FONT_HERSHEY_SIMPLEX
img_text=cv2.putText(img,'Practice',(200,250),font,1,(255,0,0),3)
# cv2.imshow("rectangle",img_rectangle) 
# cv2.imshow('Line',img_line)
# cv2.imshow('Arrowed Line',img_line_Arrowed)  
cv2.imshow('Image',img)   ###Image gets overwritten even though naming is different
k=cv2.waitKey(0)
if k==27:
    cv2.destroyAllWindows()

################################################################################

### Printing date and time on live feed
cap=cv2.VideoCapture(0)
while (cap.isOpened()):
    _,frame=cap.read()
    if _==True:
        date=str(datetime.datetime.now())
        font=cv2.FONT_HERSHEY_SIMPLEX
        frame=cv2.putText(frame,date,(50,50),font,1,(255,0,0),4)
        cv2.imshow('Live Feed',frame)
        k=cv2.waitKey(1)
        if k==27:
            break
cap.release()
cv2.destroyAllWindows()

################################################################################

### Mouse events

def click(event, x,y,flags,parameters):
    if event==cv2.EVENT_LBUTTONDOWN:
        font=cv2.FONT_HERSHEY_SIMPLEX
        text=str(x)+" "+str(y)
        cv2.putText(img,text,(x,y),font,1,(255,0,0),1)   ###Get coordinates at that point
        cv2.imshow('mouse event',img)
    if event==cv2.EVENT_RBUTTONDOWN:   ###Draw circle at that point
        
        font=cv2.FONT_HERSHEY_SIMPLEX
        b=img[y,x,0]
        g=img[y,x,1]
        r=img[y,x,2]
        channel=str(b)+" "+str(g)+" "+str(r)    ###Get color intensity at that point
        cv2.putText(img,channel,(x,y),font,0.5,(0,255,255),1)
        cv2.circle(img,(x,y),1,(0,0,255),-1)
        cv2.imshow('mouse event',img)

img=cv2.imread('photo1.png')
cv2.imshow('mouse event',img)
cv2.setMouseCallback('mouse event',click)

k=cv2.waitKey(0)
if k==27:
    cv2.destroyAllWindows()
 
 ################################################################################

###Image Additions and bitwise operators

img1=cv2.imread('photo1.png')
img2=cv2.imread('photo2.png')
img3=cv2.add(img1,img2)    ###Add 2 images
img4=cv2.addWeighted(img1,0.3,img2,0.7,0)   ###Weighted Addition
img5=cv2.bitwise_and(img1,img2)
img6=cv2.bitwise_or(img1,img2)
img7=cv2.bitwise_xor(img1,img2)
img8=cv2.bitwise_not(img1)
cv2.imshow('Image addition',img3)
cv2.imshow('Image wighted addition',img4)
cv2.imshow('Image bitwise and addition',img5)
cv2.imshow('Image bitwise or addition',img6)
cv2.imshow('Image bitwise xor addition',img7)
cv2.imshow('Image bitwise not',img8)
cv2.waitKey(0)

# ################################################################################

###Trackbar for changing BGR 
def blank(x):
    pass

img=np.zeros((500,500,3),np.uint8)
cv2.namedWindow('Image') 
cv2.createTrackbar('B','Image',0,255,blank)
cv2.createTrackbar('G','Image',0,255,blank)
cv2.createTrackbar('R','Image',0,255,blank)
cv2.createTrackbar('Switch','Image',0,1,blank) ###Controlling trackbar by conditions


while True:
    b=cv2.getTrackbarPos('B','Image')
    g=cv2.getTrackbarPos('G','Image')
    r=cv2.getTrackbarPos('R','Image')
    switch=cv2.getTrackbarPos('Switch','Image')
    if switch==1:
        img[:]=[b,g,r]
    cv2.imshow('Image',img)
    k=cv2.waitKey(1)
    
    if k==27:
        break

cv2.destroyAllWindows()

################################################################################
  
