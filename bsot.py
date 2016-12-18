#
# Simple Background Subtraction, Extraction and Object Tracking Module
#
# TODO: Make these parametric:
#       Foreground Mask Threshold: (80, 255, 0)
#       Minimum Contour Area: 50
#       Cluster Nearby Contours - Distance < 20
#       MORPH_EX Ellipse: (3, 3)
#

import cv2
from geometry import *

class Bsot:

    def __init__(self, video_url):
        self.capture = cv2.VideoCapture(video_url)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        self.fgbg = cv2.createBackgroundSubtractorMOG2()

    def has_new_frame(self):
        return self.capture.grab()

    def find_tracks(self):
        ret, self.frame = self.capture.read()
        self.fgmask = self.fgbg.apply(self.frame)
        self.fgmask = cv2.morphologyEx(self.fgmask, cv2.MORPH_OPEN, self.kernel)
        ret,thresh = cv2.threshold(self.fgmask,80,255,0)
        imC, self.contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = [contour for contour in self.contours if cv2.contourArea(contour) > 50]

        self.bounding_boxes = self.cluster_bounding_boxes(self.contours)

    def cluster_bounding_boxes(self, contours):
        bounding_boxes = []
        for i in range(len(contours)):
            x1,y1,w1,h1 = cv2.boundingRect(contours[i])

            parent_bounding_box = self.get_parent_bounding_box(bounding_boxes, i)
            if parent_bounding_box is None:
                parent_bounding_box = self.BoundingBox(Rect(x1, y1, w1, h1))
                parent_bounding_box.members.append(i)
                bounding_boxes.append(parent_bounding_box)

            for j in range(i+1, len(contours)):
                if self.get_parent_bounding_box(bounding_boxes, j) is None:
                    x2,y2,w2,h2 = cv2.boundingRect(contours[j])
                    rect = Rect(x2, y2, w2, h2)
                    distance = parent_bounding_box.rect.distance_to_rect(rect)
                    if distance < 20:
                        parent_bounding_box.update_rect(self.extend_rectangle(parent_bounding_box.rect, rect))
                        parent_bounding_box.members.append(j)
        return bounding_boxes

    def get_parent_bounding_box(self, bounding_boxes, index):
        for bounding_box in bounding_boxes:
            if index in bounding_box.members:
                return bounding_box
        return None

    def extend_rectangle(self, rect1, rect2):
        x = min(rect1.l_top.x, rect2.l_top.x)
        y = min(rect1.l_top.y, rect2.l_top.y)
        w = max(rect1.r_top.x, rect2.r_top.x) - x
        h = max(rect1.r_bot.y, rect2.r_bot.y) - y
        return Rect(x, y, w, h)

    def get_frame(self):
        return self.frame

    def get_background_image(self):
        return self.fgbg.getBackgroundImage()

    def get_foreground_image(self):
        return cv2.bitwise_and(self.frame, self.frame, mask=self.fgmask)

    def get_foreground_mask(self):
        return self.fgmask

    def get_bounding_boxes_image(self):
        bounding_box_image = self.frame.copy()
        for bounding_box in self.bounding_boxes:
            cv2.rectangle(bounding_box_image, (bounding_box.x, bounding_box.y), (bounding_box.x+bounding_box.w, bounding_box.y+bounding_box.h), bounding_box.color, 2)
        return bounding_box_image

    def destroy(self):
        cap.release()
        cv2.destroyAllWindows()

    class BoundingBox:
        def update_rect(self, rect):
            self.rect = rect
            self.x = rect.l_top.x
            self.y = rect.l_top.y
            self.w = rect.width
            self.h = rect.height
            self.color = None
            self.tracker_id = 0

        def __init__(self, rect):
            self.update_rect(rect)
            self.members = []
