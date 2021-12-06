#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
ORANGE='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
WHITE='\033[1;37m'
NC='\033[0m'

 if [[ $# -ne 2 ]] ; then
     echo -e "${WHITE}./convert.sh ${CYAN}[PATH_TO_SIF_OF_PYTORCH] ${YELLOW}[BATCH_SIZE]${NC}"
     echo -e "${WHITE}./convert.sh ${CYAN}/data/va/App/sif/pytorch_21.10-py3.sif ${YELLOW}1${NC}"
     exit 0
 fi

CONTAINER=$1
BATCH_SIZE=$2

CURRDIR=$(dirname $0)
if [ ! -d "$CURRDIR/models" ]; then
    echo -e "Path ${RED}$CURRDIR${NC} not found!"
    exit 0
fi

STARTTIME=$(date +%s)

#Convert all models
shopt -s globstar
for m in ./**/*.cfg
do
    if [ -f "$m" ]; then
        cutstart=${m:2}
        MODEL=${cutstart::-4}
        echo -e "\n"$(date +"%Y-%m-%d %T") - "${GREEN}Start building model '$MODEL'...${NC}"
        #echo "./convert.sh $CONTAINER $MODEL $BATCH_SIZE"
        ./convert.sh $CONTAINER $MODEL $BATCH_SIZE
        echo -e $(date +"%Y-%m-%d %T") - "${GREEN}Finish! '$MODEL' builded.${NC}\n"
    fi
done

ENDTIME=$(date +%s)

echo -e "${WHITE}BUILD TIME:${NC} ${ORANGE}$(($((ENDTIME-STARTTIME))/60))${NC} ${WHITE}min.${NC}"

