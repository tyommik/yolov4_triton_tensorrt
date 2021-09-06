#!/bin/bash

if [[ $# -ne 5 ]] ; then
    echo './yolo_to_trt_dynamic_batch.sh [MODEL_NAME] [INPUT_W] [INPUT_H] [NUM_CLASSES] [BATCH_SIZE]'
    exit 0
fi

MODEL=$1
WIDTH=$2
HEIGHT=$3
NUM_CLASSES=$4
BATCH_SIZE=$5
python3 yolo/yolo_to_onnx.py -m $MODEL
python3 yolo/onnx_to_tensorrt_dynamic_batch.py -m $MODEL -b $BATCH_SIZE -v -c $NUM_CLASSES --width $WIDTH --height $HEIGHT
