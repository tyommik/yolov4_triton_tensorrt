#!/bin/bash

pip install --upgrade pip
pip install onnx==1.4.1
cd /tensorrt_demos/plugins/batch && rm -f *.so *.o && make && cp libyolo_layer.so ../libyolo_layer.so && cd /tensorrt_demos/yolo