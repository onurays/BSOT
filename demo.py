#
# Demo of the BSOT library
# Usage: python demo.py video/atrium.mp4
#

import sys
import cv2
from bsot import Bsot

video_url = "video/atrium.mp4"
if len(sys.argv) > 1:
    video_url = sys.argv(1)

bsot = Bsot(video_url)

while bsot.has_new_frame():
    bsot.find_tracks()
    cv2.imshow('Input', bsot.get_frame())
    cv2.imshow('Background', bsot.get_background_image())
    cv2.imshow('Foreground',bsot.get_foreground_image())
    cv2.imshow('Foreground Mask', bsot.get_foreground_mask())
    cv2.imshow('Bounding Boxes', bsot.get_bounding_boxes_image())

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        bsot.destroy()
        break
