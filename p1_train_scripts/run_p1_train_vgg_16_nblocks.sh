#!/bin/bash
#
# This script performs the following operations:
set -e

# Where the fine-tuned ResNetV1-50 checkpoint is saved to.
FINETUNED_CHECKPOINT_DIR=$MY_PROJECT_HOME/tmp/cub200-models/vgg_16_glb_only/train_all/model.ckpt-40000

# Where the dataset is saved to.
DATASET_DIR=$MY_PROJECT_HOME/tmp/cub200/train_test_split

# input variables 
for kp in 0.3 0.5 0.7
do  

learning_rate=0.000005 #$2 # default is: 0.000001
block_size=2

# Where the training checkpoint and logs will be saved to.
TRAIN_DIR=$MY_PROJECT_HOME/tmp/cub200-models/prune_vgg_16/localtrain_lr${learning_rate}_bs${block_size}


python p1_train_vgg_16_nblocks.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=cub200 \
  --train_dataset_name=train \
  --test_dataset_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=vgg_16 \
  --checkpoint_path=${FINETUNED_CHECKPOINT_DIR} \
  --max_number_of_steps=5000 \
  --batch_size=32 \
  --test_batch_size=50 \
  --learning_rate=${learning_rate} \
  --learning_rate_decay_type=fixed \
  --log_every_n_steps=100 \
  --summary_every_n_steps=500 \
  --evaluate_every_n_steps=500 \
  --runmeta_every_n_steps=13600 \
  --optimizer=sgd \
  --weight_decay=0.0001 \
  --kept_percentages=${kp} \
  --max_to_keep=1 \
  --block_size=${block_size} \
  --block_config_id=0 \
  --continue_training=False \
  --last_conv_pruned=False 

done 

