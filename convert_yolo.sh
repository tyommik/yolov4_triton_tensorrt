#!/bin/bash

# singularity exec --nv  --no-home --bind ../yolov4_triton_tensorrt:/tensorrt_demos --writable-tmpfs pytorch_21.06-py3.sif bash -c 'cp cfg/pip.conf /etc/ && cd /tensorrt_demos/ && ./yolov4.sh '"$MODEL"' --verbose'
singularity exec --nv --bind ../yolov4_triton_tensorrt:/tensorrt_demos --writable-tmpfs  --sandbox   pytorch_21.06-py3.sif bash -c 'cd /tensorrt_demos/ && ./yolov4.sh '"$MODEL"' --verbose'
# singularity exec --nv --no-home --bind ../yolov4_triton_tensorrt:/tensorrt_demos --writable-tmpfs pytorch_21.06-py3.sif bash
