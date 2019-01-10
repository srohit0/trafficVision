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
#      inference the yolo v2 model in openVX format.
#
import numpy as np
import cv2

import yoloOpenVX
import speedEstimator

class yoloInferenceNet:
    def __init__(self, weights):
        # create inference graph.
        self.handle =  yoloOpenVX.interface.annCreateInference(weights.encode('utf-8'))

        # get graph details for adapting the input.
        input_info,output_info = yoloOpenVX.interface.annQueryInference().decode("utf-8").split(';')
        self.ni,self.ci,self.hi,self.wi = map(int, input_info.split(',')[2:])

        self.frame   = np.empty(0)
        self.offsetX = 0;
        self.offsetY = 0;
        self.scale   = 1;

        self.estimator = speedEstimator.estimate()

    # print details
    def __repr__(self):
        return "%s <ni:%d wi:%d hi:%d scale:%f offsetX:%d offsetY:%d>" % (self.__class__.__name__, self.ni, self.wi, self.hi, self.scale, self.offsetX, self.offsetY)

    # compute scale factor to convert original
    # frame to yonoNet frame.
    def scaleFactor (self, ow, oh, nw, nh):
        h_ratio = float(nh)/oh
        w_ratio = float(nw)/ow
        self.scale   = 1.0
        self.offsetX = 0
        self.offsetY = 0

        if (h_ratio>w_ratio):
            self.scale   = float(nw)/ow;
            self.offsetY = int((nh-float(oh*self.scale))/2.0)
        else:
            self.scale   = float(nh)/oh;
            self.offsetX = int((nw-float(ow*self.scale))/2.0)

        return (self.scale, int(self.offsetX), int(self.offsetY))

    # scale frame to yoloNet frame
    def yoloInput(self, frame):

        self.frame = frame; # (1080, 1920 ,3)

        # scale frame to yolo size of 416, 416, 3
        self.scaleFactor(frame.shape[1], frame.shape[0], self.wi, self.hi)

        new_frame = np.zeros((self.hi,self.wi,self.ci), dtype=np.uint8)
        new_frame[self.offsetY:self.hi-self.offsetY, self.offsetX:self.wi-self.offsetX, :] = \
                cv2.resize(frame.copy(), None, fx=self.scale, fy=self.scale, \
                        interpolation=cv2.INTER_CUBIC)

        return new_frame

    # add boxes around detected objects.
    def addBoxes(self, frame, boxes):
        dup_frame = frame.copy()
        if boxes == None:
            return dup_frame

        boxes = self.estimator.speed(boxes)
        for one_box in boxes:
            left, top, right, bottom, confidence, ilabel, label, speed, bdir = one_box
            if ( frame.shape == self.frame.shape ):
                # scale to original frame.
                left   = int((left-self.offsetX)/self.scale)
                top    = int((top-self.offsetY)/self.scale)
                right  = int((right-self.offsetX)/self.scale)
                bottom = int((bottom-self.offsetY)/self.scale)
            confidence = int(confidence * 100)
            if confidence > 20:
                color = self.estimator.color(speed);
                #print (top+bottom)/2.0, speed, bdir, (color);
                cv2.rectangle(dup_frame, (left,top), (right,bottom), color, thickness=5)
                size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                #width = size[0][0] + 50
                width = abs(right-left-5)
                height = size[0][1]
                text = str(int(confidence)) + "% " + label
                if ( speed > 0 ):
                    text = str(int(speed)) + "mph " + label
                cv2.rectangle(dup_frame, (left+5, (bottom-5) - (height+5)), ((left + width), (bottom-5)),(255,0,0),-1)
                cv2.putText(dup_frame,text,((left + 5),(bottom-10)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
        return dup_frame

    def destroy(self):
        yoloOpenVX.interface.annReleaseInference(self.handle)


