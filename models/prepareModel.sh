#!/bin/bash -xvf
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
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


hash curl 2>/dev/null ||
    {
        echo >&2 "ERROR: Curl is required to download yolo model. Install curl with command below and try again."
        echo >&2 "sudo apt-get install curl"
        exit 1;
    }

hash python 2>/dev/null ||
    {
        echo >&2 "ERROR: Python is required to run traffic Vision app. Install python with command below and try again."
        echo >&2 "sudo apt-get install python"
        exit 1;
    }

hash runvx 2>/dev/null ||
    {
        echo >&2 "ERROR: MIVisionX is required to run traffic Vision app. Install MIVisionX and try again."
        echo >&2 "git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX"
        exit 1;
    }


###################################################
# Step 1: Download yolo model
###################################################
mkdir yolomodels && cd yolomodels
curl https://pjreddie.com/media/files/yolov2-tiny-voc.weights -o yolov2-tiny-voc.weights
curl https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2-tiny-voc.cfg -o yolov2-tiny-voc.cfg
cd ..

###################################################
# Step 2: Convert to yolo model to caffe model
###################################################

mkdir caffemodels
filename=yolov2-tiny-voc

yolocfg=./yolomodels/$filename.cfg
yoloweight=./yolomodels/$filename.weights
yolocfgcaffe=./caffemodels/$filename.prototxt
yoloweightcaffe=./caffemodels/$filename.caffemodel

if [[ -f $yolocfg && -f $yoloweight ]]; then
    echo "Found yolo model: $yolocfg , $yoloweight"
else
    echo "Could not locate yolo model: $yolocfg , $yoloweight"
    exit 1
fi

echo "Convert yolo to caffe"
python create_yolo_prototxt.py $yolocfg $yolocfgcaffe
python create_yolo_caffemodel.py -m $yolocfgcaffe -w $yoloweight -o $yoloweightcaffe


###################################################
# Step 3: Convert caffe model to NNIR format
###################################################
mkdir nnirModel
python /opt/rocm/mivisionx/model_compiler/python/caffe2nnir.py caffemodels/$filename.caffemodel ./nnirModel --input-dims 1,3,416,416

###################################################
# Step 4: Convert NNIR to OpenVX format
###################################################
python /opt/rocm/mivisionx/model_compiler/python/nnir2openvx.py ./nnirModel ./openVXModel

###################################################
# Step 5: Compile openVX model
###################################################
cd ./openVXModel
mkdir build && cd build
cmake ..
make
cd ../..
cp ./openVXModel/build/libannmodule.so ../lib
cp ./openVXModel/build/libannpython.so ../lib

