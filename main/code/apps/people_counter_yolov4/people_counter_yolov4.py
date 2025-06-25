import cv2
import numpy as np
from threading import Thread # Create and manage threads in program
import time

from centroidtracker import CentroidTracker
from trackableobject import TrackableObject
import dlib
#import imutils
#from imutils.video import VideoStream
#from imutils.video import FPS

import timeit

from shapely.geometry import Point, Polygon

import warnings # ignore the warning message
warnings.filterwarnings("ignore", category=UserWarning)

FW_version = "PCC_0.1"

min_confidence = 0.5
weight_file = "model/people_v4tiny/people-v4tiny_best.weights"
cfg_file = "model/people_v4tiny/people-v4tiny.cfg"
name_file = "model/people_v4tiny/people-v4tiny.names"


# Load Yolo
net = cv2.dnn.readNet(weight_file, cfg_file)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA) # set the perferred backend for running the model
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA) # Cuda backend should be used

classes = []
with open(name_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
#output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

colors = np.random.uniform(0,255,size=(len(classes),3)) # Generate a NumPy array of random values
use_cuda = 1

try :
    if use_cuda:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    else :
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

ct = CentroidTracker(maxDisappeared=50, maxDistance=800)
trackers = []
trackableObjects = {}

class work(object):
    def __init__(self, src=0):
        # Create a VideoCapture object
        self.capture = cv2.VideoCapture(src)

        #Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

        self.totalFrames = 0
        self.totalLeft = 0
        self.totalRight = 0

        self.skip_frames = 60
        self.W = None
        self.H = None

        self.p1 = Point(1290,70)
        self.p2 = Point(810,830)

        self.coords = [(self.p1.x-20, self.p1.y), (self.p2.x-20, self.p2.y), (self.p2.x+20, self.p2.y), (self.p1.x+20, self.p1.y)]
        self.poly = Polygon(self.coords)

        #self.pts = np.array([[self.p1.x-20,self.p1.y],[self.p2.x-20,self.p2.y],[self.p2.x+20,self.p2.y],[self.p1.x+20,self.p1.y]],np.int32)
        #self.pts = self.pts.reshape((-1, 1, 2))

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                # Frame read
                (self.status, self.frame) = self.capture.read()
                if not self.status:
                    break
            else :
                print('Recording is now finished')
                self.capture.release()
                cv2.destroyAllWindows()
                exit(1)

            time.sleep(.03)

    def show_frame(self):

        frame_copy = self.frame.copy()

        if self.W is None or self.H is None:
            (self.H, self.W) = frame_copy.shape[:2]

        start_t = timeit.default_timer()

        self.status = "Waiting"
        self.rects = []

        if (self.totalFrames % self.skip_frames) == 0:
            self.status = "Detecting"
            self.trackers = []

            # YOLO
            height, width, _ = frame_copy.shape
            blob = cv2.dnn.blobFromImage(frame_copy, 0.00392, (512,512), (0,0,0), True, crop=False)

            net.setInput(blob)
            outs = net.forward(output_layers)
            class_ids = []
            confidences = []
            boxes = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > min_confidence :
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        # coordinate
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x,y,w,h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.5)

            for i in range(len(boxes)):
                if i in indexes:
                    x,y,w,h = boxes[i]
                    self.tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(x, y, x+w, y+h)
                    self.tracker.start_track(frame_copy, rect)
                    self.trackers.append(self.tracker)
        else :
            # loop over the trackers
            for self.tracker in self.trackers:
                # set the status of our system to be 'tracking' rather
                # than 'waiting' or 'detecting'
                self.status = "Tracking"

                # update the tracker and grab the updated position
                self.tracker.update(frame_copy)
                pos = self.tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                # add the bounding box coordinates to the rectangles list
                self.rects.append((startX, startY, endX, endY))

        cv2.line(frame_copy, (int(self.p1.x), int(self.p1.y)), (int(self.p2.x), int(self.p2.y)), (0,255,255), 2)
        #cv2.polylines(frame_copy, [self.pts], isClosed=True, color=(0, 255, 0), thickness=1)

        self.objects = ct.update(self.rects)

        # loop over the tracked objects
        for (objectID, centroid) in self.objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            to = trackableObjects.get(objectID, None)
            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                x = [c[0] for c in to.centroids]
                direction_x = centroid[0] - np.mean(x)
                y = [c[1] for c in to.centroids]
                direction_y = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                # check to see if the object has been counted or not
                if not to.counted:
                    point = Point(centroid[0], centroid[1])
                    within = point.within(self.poly)

                    if direction_x > 0 and direction_y < 0 and within == True:
                        self.totalRight += 1
                        to.counted = True
                    elif direction_x > 0 and direction_y == 0 and within == True:
                        self.totalRight += 1
                        to.counted = True
                    elif direction_x > 0 and direction_y > 0 and within == True:
                        self.totalRight += 1
                        to.counted = True
                    elif direction_x == 0 and direction_y > 0 and within == True:
                        self.totalRight += 1
                        to.counted = True
                    elif direction_x == 0 and direction_y < 0 and within == True:
                        self.totalLeft += 1
                        to.counted = True
                    elif direction_x < 0 and direction_y > 0 and within == True:
                        self.totalLeft += 1
                        to.counted = True
                    elif direction_x < 0 and direction_y == 0 and within == True:
                        self.totalLeft += 1
                        to.counted = True
                    elif direction_x < 0 and direction_y < 0 and within == True:
                        self.totalLeft += 1
                        to.counted = True

            # store the trackable object in our dictionary
            trackableObjects[objectID] = to

            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frame_copy, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame_copy, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        terminate_t = timeit.default_timer()
        FPS = int(1./(terminate_t - start_t ))

        info = [
            ("Right", self.totalRight),
            ("Left", self.totalLeft),
            ("Status", self.status),
            ("FPS", FPS),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame_copy, text, (10, self.H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # create a window named "Frame"
        window_title = "Frame"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

        # show the output frame
        cv2.imshow(window_title, frame_copy)

        # resize the window to the desired width and height
        new_width, new_height = 1280, 720
        cv2.resizeWindow(window_title, new_width, new_height)

        self.totalFrames += 1
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)

if __name__ == "__main__":

    print("People Counter running!! ver:{}".format(FW_version))

    file_name = "video/test1.mp4"

    video_stream = work(file_name)

    while True:
        try:
            video_stream.show_frame()
        except AttributeError:
            pass