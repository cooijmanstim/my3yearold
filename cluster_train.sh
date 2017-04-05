#!/bin/bash

if [ -z "$MSCOCO_TFRECORD_DIR" ]
then
    echo "MSCOCO_TFRECORD_DIR not set"
    exit 1
fi

if [ -z "$MSCOCONET_HYPERPARAMETERS" ]
then
    echo "MSCOCONET_HYPERPARAMETERS not set"
    exit 1
fi

source /rap/jvb-000-aa/stack/.bashrc

python $HOME/repos/my3yearold/train.py \
       --data_dir=$LSCRATCH/cooijmat/mscoconet \
       --base_output_dir=$PWD \
       --hp="$MSCOCONET_HYPERPARAMETERS"
       "$@"
