from __future__ import print_function
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
#where is the haarcascades?
parser.add_argument('--face_cascade', help='Path to face cascade.', default='C:/openCV/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='C:/openCV/opencv/build/etc/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--body_cascade', help='Path to face cascade.', default='C:/openCV/opencv/build/etc/haarcascades/haarcascade_upperbody.xml')

#camera usbID
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
body_cascade_name = args.body_cascade
face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()
body_cascade = cv.CascadeClassifier()

if not body_cascade.load(cv.samples.findFile(body_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)

camera_device = args.camera 

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.bodyColor = [177, 177 , 177]
        self.faceColor = [144, 144, 144]
        self.eyeColor = [144, 255, 144]
        self.video = cv.VideoCapture(camera_device)
        if not self.video.isOpened:
            print('--(!)Error opening video capture')
            exit(0)
    
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        frame_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        frame_gray = cv.equalizeHist(frame_gray)
        #-- Detect faces
        faces = face_cascade.detectMultiScale(frame_gray)
        bodys = body_cascade.detectMultiScale(frame_gray)
        #faceColor = [144, 144, 144]
        #eyeColor = [144, 255, 144]
        for (x1,y1,w1,h1) in bodys :
            cv.rectangle(image, (x1, y1),  
                      (x1 + h1, y1 + w1),  
                      (0, 255, 0), 5) 
        for (x,y,w,h) in faces:
            center = (x + w//2, y + h//2)
            image = cv.ellipse(image, center, (w//2, h//2), 0, 0, 360, self.faceColor, 4)
            faceROI = frame_gray[y:y+h,x:x+w]
            #-- In each face, detect eyes
            eyes = eyes_cascade.detectMultiScale(faceROI)
            for (x2,y2,w2,h2) in eyes:
                eye_center = (x + x2 + w2//2, y + y2 + h2//2)
                radius = int(round((w2 + h2)*0.25))
                image = cv.circle(image, eye_center, radius, self.eyeColor, 4)



        ret, jpeg = cv.imencode('.jpg', image)
        return jpeg.tobytes()