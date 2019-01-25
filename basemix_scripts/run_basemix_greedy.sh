#!/bin/bash
# This script globally train a pruned network
set -e

# Where the fine-tuned ResNetV1-50 checkpoint is saved to.
FINETUNED_CHECKPOINT_DIR=$MY_PROJECT_HOME/tmp/cub200-models/resnet_v1_50_glb_only/train_all/model.ckpt-40000

# manually set the checkpoint dir to use 
#FINETUNED_CHECKPOINT_DIR=/home/hguan2/tensorflow/hg_slim/tmp/cub200-models/prune_lin03/exp4/basemix_greedy_lr0.001_1/0.6_0.7_1.0_1.0/train

# Where the dataset is saved to.
DATASET_DIR=$MY_PROJECT_HOME/tmp/cub200/train_test_split

# the dataset to use 
DATASET_NAME=cub200

# input variables
stage_id=0
option_id=0
drop_rate=0.01 
learning_rate=0.001
# Where the training (fine-tuned) checkpoint and logs will be saved to.

TRAIN_DIR=$MY_PROJECT_HOME/tmp/cub200-models/prune/basemix_greedy_dr${drop_rate}

python basemix_greedy.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET_NAME} \
  --train_dataset_name=train \
  --test_dataset_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=resnet_v1_50 \
  --checkpoint_path=${FINETUNED_CHECKPOINT_DIR} \
  --max_number_of_steps=30000 \
  --batch_size=32 \
  --test_batch_size=100 \
  --learning_rate=${learning_rate} \
  --learning_rate_decay_factor=0.1 \
  --learning_rate_decay_type=exponential \
  --num_epochs_per_decay=60 \
  --log_every_n_steps=100 \
  --summary_every_n_steps=500 \
  --evaluate_every_n_steps=500 \
  --runmeta_every_n_steps=55500 \
  --optimizer=sgd \
  --weight_decay=0.00001 \
  --stage_index=${stage_id} \
  --option_index=${option_id} \
  --baseline_accuracy=0.77 \
  --continue_training=True \
  --max_to_keep=1

 


