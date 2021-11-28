#!/bin/bash

CONTAINER="pytorch_21.10-py3.sif"

 if [[ $# -ne 2 ]] ; then
     echo './convert.sh [MODEL_NAME] [BATCH_SIZE]'
     echo './convert.sh models/yolov4-heads 0'
     echo 'BATCH_SIZE = [1,2,4,6,...] for STATIC BATCH'
     echo 'BATCH_SIZE = 0 for DYNAMIC BATCH'
     exit 0
 fi

# Parse width, height and classes params from model config.
commandOutput="$(python3 yolo/parse_config.py  $1)"
stringarray=($commandOutput)

MODEL=$1
WIDTH=${stringarray[0]}
HEIGHT=${stringarray[1]}
NUM_CLASSES=${stringarray[2]}
BATCH_SIZE=$2

echo "============= YOLO PARAMETERS ============="
echo "Model width ${stringarray[0]}"
echo "Model height: ${stringarray[1]}"
echo "Model classes ${stringarray[2]}"

if [[ $BATCH_SIZE -eq 0 ]]
then
  echo "Convert Dynamic batching ..."
  singularity exec --nv --bind ../yolov4_triton_tensorrt:/tensorrt_demos --writable-tmpfs --contain $CONTAINER bash -c 'cd /tensorrt_demos/ && ./yolo_to_trt_dynamic_batch.sh '"$MODEL"' '"$WIDTH"' '"$HEIGHT"' '"$NUM_CLASSES"' '"$BATCH_SIZE"' --verbose'
else
  echo "Convert Static batch=$BATCH_SIZE ..."
  singularity exec --nv --bind ../yolov4_triton_tensorrt:/tensorrt_demos --writable-tmpfs --contain $CONTAINER bash -c 'cd /tensorrt_demos/ && ./yolo_to_trt_static_batch.sh '"$MODEL"' '"$WIDTH"' '"$HEIGHT"' '"$NUM_CLASSES"' '"$BATCH_SIZE"' --verbose'
fi
