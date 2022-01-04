import cv2
import numpy as np

def Pseudocoloring(r,g,b,highest,lowest):
    range=highest-lowest
    range=np.uint8(range)
    r_temp=r.copy()
    g_temp=g.copy()
    b_temp=b.copy()

    r[r_temp<range/2]=0
    r[r_temp>=range/2]=255*(r[r_temp>=range/2]- range/2)/(3*range/4 - range/2)
    r[r_temp>=3*range/4]=255
    
    g[g_temp<range/4]=255*(g[g_temp<range/4]-lowest)/(range/4-lowest)
    g[g_temp>=range/4]=255
    g[g_temp>=3*range/4 ]=255- 255*(g[g_temp>=3*range/4 ]-3*range/4)/(range/4)

    
    b[b_temp<range/4]=255
    b[b_temp>=range/4]=255- 255*(b[b_temp>=range/4]-range/4)/(range/4)
    b[b_temp>=range/2]=0
    img=cv2.merge([b,g,r])
    
    return img



image_name=input('Enter Input Image Name: ')
img=cv2.imread(image_name,0)

lowest=np.min(img)
highest=np.max(img)
range=highest-lowest
img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
r,g,b=cv2.split(img)
img_converted=Pseudocoloring(r,g,b,highest,lowest)

cv2.imshow('Pesudo colored Image',img_converted)
cv2.imshow('Original Image',img)
k=cv2.waitKey(0)
if k==27:
    cv2.destroyAllWindows()
