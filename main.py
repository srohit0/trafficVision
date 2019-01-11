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

# Description:
#      Detect Vehicle using AMD MIVisionX Inferencing Engine.
#
from __future__ import print_function
import os, time
import argparse
import numpy as np
import cv2

import yoloOpenVX
import inference

if __name__ == '__main__' :

    parseHandle = argparse.ArgumentParser(description=
            'Detect Vehicles in static video or a live feed.')
    parseHandle.add_argument('--video', dest='video',
            type=str, default="./media/demo.mp4",
            help='path to video file.')
    parseHandle.add_argument('--cam_ip', dest='cam_ip', type=str,
            default='', help='IP address for video cam.')

    args    = parseHandle.parse_args()

    if ( args.cam_ip  ) :
        # must be a mjpg or h264 streaming
        window_title = args.cam_ip + "- AMD Object Detection on Live Feed"
        feed         = args.cam_ip
    elif ( args.video ) :
        window_title = os.path.basename(args.video) + "- AMD Object Detection on Recorded Feed"
        feed         = args.video
    else:
        print ("Error: no video source.");
        exit(0);

    yoloNet = inference.yoloInferenceNet(yoloOpenVX.weights);
    cv2.namedWindow(window_title, cv2.WINDOW_GUI_EXPANDED)
    cap = cv2.VideoCapture(feed)

    if ( cap.isOpened() == False ):
        print ("Error: could not open video feed", feed);

    iframe = 0;
    while(cap.isOpened()):

        ret, frame    = cap.read()
        if ret == False:
            break;

         if ( iframe == 0 ):
             cv2.resizeWindow(window_title, frame.shape[1], frame.shape[0])

        iframe = iframe + 1
        resized_frame = yoloNet.yoloInput(frame); # image to display
        frame_array   = np.concatenate((resized_frame[:,:,0], resized_frame[:,:,1], resized_frame[:,:,2]), 0)
        boxes         = yoloOpenVX.model.detectBoxes(yoloNet.handle, np.ascontiguousarray(frame_array, dtype=np.float32)/(255.0))

        frame_w_boxes = yoloNet.addBoxes(frame, boxes);

        cv2.imshow(window_title, frame_w_boxes)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        if ( os.path.splitext(args.cam_ip)[1] == ".jpg"  ) :
            cap = cv2.VideoCapture(feed); # required for static jpg like stream in Bosch cams
        #time.sleep(5)

    cap.release()
    cv2.destroyAllWindows()
    print ("Processed a total of ", iframe, "frames");

    yoloNet.destroy();



