#!/bin/bash

MODEL=$1
WIDTH=$2
HEIGHT=$3
NUM_CLASSES=$4
BATCH_SIZE=$5
CONTAINER_VER=$6

cd /tensorrt_demos/plugins/batch && rm -f *.so *.o && make && cp libyolo_layer.so ../libyolo_layer.so && cd /tensorrt_demos
python3 yolo/yolo_to_onnx.py -m $MODEL
python3 yolo/onnx_to_tensorrt_dynamic_batch.py -m $MODEL -b $BATCH_SIZE -v -c $NUM_CLASSES --width $WIDTH --height $HEIGHT --ver $CONTAINER_VER
