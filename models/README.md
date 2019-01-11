
## Model Conversion
```
% ./prepareModel.sh
```


<img src="../media/speed_detection_model_conversion.jpg" width=680>

## Pre-Requisites

1. curl to download yolo model. ```sudo apt-get install curl```
1. [Caffe](http://caffe.berkeleyvision.org/installation.html)
      > Make sure to update PYTHONPATH, ```export PYTHONPATH=${CAFFE_ROOT}/python:$PYTHONPATH``` after installation.
1. [AMD's MIVisionX toolkit](https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/)

## Model Conversion Steps
These steps are included in prepareModel.sh. Repeated here for clarification:

```
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

echo "Convert yolo to caffe"
python create_yolo_prototxt.py $yolocfg $yolocfgcaffe
python create_yolo_caffemodel.py -m $yolocfgcaffe -w $yoloweight -o $yoloweightcaffe


###################################################
# Step 3: Convert caffe model to NNIR format
###################################################
mkdir /nnirModel
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
```
