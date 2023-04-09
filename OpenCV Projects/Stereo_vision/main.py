import numpy as np
import cv2
import matplotlib.pyplot as plt


if __name__ == '__main__':
    imL = cv2.imread("left.png")
    imR = cv2.imread("right.png")

    stereo = cv2.StereoSGBM_create(minDisparity=16,
                                    numDisparities=96,
                                    blockSize=5,
                                    uniquenessRatio=10,
                                    speckleWindowSize=100,
                                    speckleRange=2,
                                    disp12MaxDiff=0,
                                    P1=200,
                                    P2=800,)
    disparity = stereo.compute(imL, imR)

    plt.imsave("disparity_map.png", disparity, cmap='gray')
    plt.imshow(disparity, 'gray')
    plt.show()

    im_shape = disparity.shape
    x, y = np.meshgrid(np.arange(im_shape[1]), np.arange(im_shape[0]))

    imL_color = cv2.cvtColor(imL, cv2.COLOR_BGR2RGB)

    disparity_flat = np.where(disparity < 0, 0, disparity)
    point_cloud = np.dstack((x, y, disparity_flat, imL_color)).reshape(-1, 6)

    header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    
    f = open("points.ply", "a")
    f.write(header % dict(vert_num=point_cloud.shape[0]))
    f.write("\n")
    for pc in point_cloud:
        line = line = f"{float(pc[0])} {float(pc[1])} {float(pc[2]/2)} {pc[3]} {pc[4]} {pc[5]}\n"
        f.write(line)
    f.close()
   