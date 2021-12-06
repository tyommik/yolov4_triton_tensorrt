#!/bin/bash

#Example: ./convert.sh /data/va/App/sif/pytorch_21.10-py3.sif models/heads_YOLO4_704/10_08_2021/yolov4-heads 16

RED='\033[0;31m'
GREEN='\033[0;32m'
ORANGE='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
WHITE='\033[1;37m'
NC='\033[0m'

 if [[ $# -ne 3 ]] ; then
     echo './convert.sh [PATH_TO_SIF] [MODEL_NAME] [BATCH_SIZE]'
     echo './convert.sh pytorch_21.10-py3.sif models/yolov4-heads 0'
     echo 'BATCH_SIZE = [1,2,4,6,...] for STATIC BATCH'
     echo 'BATCH_SIZE = 0 for DYNAMIC BATCH'
     exit 0
 fi

CONTAINER=$1 #"/data/va/App/sif/pytorch_21.10-py3.sif"
MODEL=$2
BATCH_SIZE=$3

if [ ! -f "$CONTAINER" ]; then
    echo -e "Path ${RED}$CONTAINER${NC} to file of sif container not found!"
    exit 0
fi

if [ ! -f "$MODEL.cfg" ]; then
    echo -e "Path ${RED}$MODEL.cfg${NC} to cfg file of model not found!"
    exit 0
fi

CONTAINER_VER=$(echo $CONTAINER| grep -o -E '[0-9]+[\.][0-9]+')

echo -e "\n============ SCRIPT PARAMETERS ============"
echo -e "${ORANGE}SIF PATH:${NC}   ${WHITE}$CONTAINER${NC}"
echo -e "${ORANGE}MODEL PATH:${NC} ${WHITE}$MODEL${NC}"
echo -e "${ORANGE}BATCH SIZE:${NC} ${WHITE}$BATCH_SIZE${NC}"
echo -e "${ORANGE}TRT VER:${NC}    ${WHITE}$CONTAINER_VER${NC}\n"


# Parse width, height and classes params from model config.
commandOutput="$(python3 yolo/parse_config.py $MODEL)"
stringarray=($commandOutput)

WIDTH=${stringarray[0]}
HEIGHT=${stringarray[1]}
NUM_CLASSES=${stringarray[2]}

echo "============= YOLO PARAMETERS ============="
echo -e "${CYAN}Model width:${NC}   ${WHITE}${stringarray[0]}${NC}"
echo -e "${CYAN}Model height:${NC}  ${WHITE}${stringarray[1]}${NC}"
echo -e "${CYAN}Model classes:${NC} ${WHITE}${stringarray[2]}${NC}\n"

if [[ $BATCH_SIZE -eq 0 ]]
then
  echo -e "${PURPLE}Convert dynamic batching ...${NC}\n"
  singularity exec --nv --bind ./:/tensorrt_demos --writable-tmpfs --contain $CONTAINER bash -c 'cd /tensorrt_demos/ && ./yolo_to_trt_dynamic_batch.sh '"$MODEL"' '"$WIDTH"' '"$HEIGHT"' '"$NUM_CLASSES"' '"$BATCH_SIZE"' '"$CONTAINER_VER"' --verbose'
else
  echo -e "${PURPLE}Convert static batch = ${WHITE}$BATCH_SIZE${NC} ${PURPLE}...${NC}\n"
  echo "singularity exec --nv --bind ../yolov4_triton_tensorrt:/tensorrt_demos --writable-tmpfs --contain $CONTAINER bash -c 'cd /tensorrt_demos/ && ./yolo_to_trt_static_batch.sh '"$MODEL"' '"$WIDTH"' '"$HEIGHT"' '"$NUM_CLASSES"' '"$BATCH_SIZE"' --verbose'"
  singularity exec --nv --bind ./:/tensorrt_demos --writable-tmpfs --contain $CONTAINER bash -c 'cd /tensorrt_demos/ && ./yolo_to_trt_static_batch.sh '"$MODEL"' '"$WIDTH"' '"$HEIGHT"' '"$NUM_CLASSES"' '"$BATCH_SIZE"' '"$CONTAINER_VER"' --verbose'
fi
