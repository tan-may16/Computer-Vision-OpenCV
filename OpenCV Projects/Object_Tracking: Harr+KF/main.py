import cv2
import numpy as np
import argparse


def detect_bbox(frame, classifier):
    frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=classifier.detectMultiScale(frame_gray,1.1,4)
    if len(faces) > 0:
        return faces[0] # Returning only 1 bbox
    return None
    

def create_system(s,m,A,C,Q,R):
    KF = cv2.KalmanFilter(s, m)
    KF.transitionMatrix = A.astype(np.float32)
    KF.measurementMatrix = C.astype(np.float32)

    KF.processNoiseCov = Q.astype(np.float32)
    KF.measurementNoiseCov = R.astype(np.float32)
    
    return KF

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Define Parameterss')
    
    parser.add_argument('--classifier_path', type=str, default = 'haarcascade_frontalface_default.xml') 
    parser.add_argument('--states', type=int, default = 6) # (x, y, w, h, vx, vy)
    parser.add_argument('--measurements', type=int, default = 4) # mx, my, mw, mh
    parser.add_argument('--A', type=lambda s: np.reshape(np.fromstring(s, sep=','), (-1, 2)), default= np.array([[1,0,0,0,1,0], [0,1,0,0,0,1], [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]], np.float32)) 
    parser.add_argument('--C', type=lambda s: np.reshape(np.fromstring(s, sep=','), (-1, 2)), default= np.array([[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0]], np.float32)) 
    parser.add_argument('--Q', type=lambda s: np.reshape(np.fromstring(s, sep=','), (-1, 2)), default= np.eye(6).astype(np.float32)) 
    parser.add_argument('--R', type=lambda s: np.reshape(np.fromstring(s, sep=','), (-1, 2)), default= 0.5*np.eye(4).astype(np.float32)) 
    
    
    args = parser.parse_args()
    
    KF = create_system(args.states,args.measurements,args.A,args.C,args.Q,args.R)
    
    classifier=cv2.CascadeClassifier(args.classifier_path)
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    
    # Initial Bbox
    bbox = cv2.selectROI(frame, False)
    x, y, w, h = bbox
    cv2.destroyAllWindows()
    
    #Kalman Filter Loop
    while True:
        
        ret, frame = cap.read()
        prediction = KF.predict()
        
        objects = detect_bbox(frame, classifier)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        
        # x, y, w, h = bbox
        if objects is not None: 
            x,y,w,h = objects
            
        else:
            print("Object Not in frame")
            
        object_center = x + w/2,y + h/2
        KF.correct(np.array([[object_center[0]], [object_center[1]], [w], [h]], np.float32))
        new_bbox = (int(prediction[0] - prediction[2]/2), int(prediction[1] - prediction[3]/2), int(prediction[2]), int(prediction[3]))
        
        cv2.rectangle(frame, (new_bbox[0], new_bbox[1]), (new_bbox[0] + new_bbox[2], new_bbox[1] + new_bbox[3]), (0, 255, 0), 2)
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()


