#!/usr/bin/python
#
# Copyright (c) 2018. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in

import sys, os
import cv2

if ( len(sys.argv) < 2 ):
    print "\nUsage   : ", sys.argv[0], "<cam_ip_for_streaming>\n"
    print "examples: ", sys.argv[0], "\'http://166.149.104.112:8082/snap.jpg\'"
    print "          ", sys.argv[0], "\'http://177.72.7.85:8001/mjpg/video.mjpg\'\n"
    exit(0);

cap = cv2.VideoCapture(sys.argv[1])
if ( cap.isOpened() ):
    print "opened stream"
else:
    print "failed to open stream"
    exit(0)

cv2.namedWindow (sys.argv[1], cv2.WINDOW_GUI_EXPANDED)
cv2.resizeWindow(sys.argv[1], 480, 270)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break

    cv2.imshow(sys.argv[1], frame)

    key = cv2.waitKey(1)
    if ( os.path.splitext(sys.argv[1])[1] == ".jpg"  ) :
        cap = cv2.VideoCapture(sys.argv[1]); # required for static jpg like stream in Bosch cams
    if key & 0xFF == ord('q'):
        break

