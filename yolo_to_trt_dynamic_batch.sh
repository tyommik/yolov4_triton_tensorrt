#!/bin/bash

MODEL=$1
python3 yolo/yolo_to_onnx.py -m $MODEL
python3 yolo/onnx_to_tensorrt_dynamic_batch.py -m $MODEL -b 16 -v -c 1
