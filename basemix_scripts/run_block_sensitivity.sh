#!/bin/bash
# This script test the sensitivity of each block in resnet to pruning 

set -e
# Where the fine-tuned ResNetV1-50 checkpoint is saved to.
FINETUNED_CHECKPOINT_DIR=$MY_PROJECT_HOME/tmp/cub200-models/resnet_v1_50_glb_only/train_all/model.ckpt-40000

# Where the dataset is saved to.
DATASET_DIR=$MY_PROJECT_HOME/tmp/cub200/train_test_split

# # input variables
# block_id=$1 
# kp=$2

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=$MY_PROJECT_HOME/tmp/cub200-models/prune/block_sensitivity

for block_id in `seq 0 15`
do 
  for kp in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
  do 
echo "block_id=${block_id}, kp=${kp}"
python block_sensitivity.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=cub200 \
  --train_dataset_name=train \
  --test_dataset_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=resnet_v1_50 \
  --checkpoint_path=${FINETUNED_CHECKPOINT_DIR} \
  --batch_size=32 \
  --test_batch_size=100 \
  --weight_decay=0.00001 \
  --kept_percentage=${kp} \
  --block_id=${block_id}

 
done 
done 

