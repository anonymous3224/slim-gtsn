#!/bin/bash
#
# This script performs the following operations:
# locally train pruned blocks on the computer:
echo $COMPUTER_NAME
set -e

# Where the fine-tuned ResNetV1-50 checkpoint is saved to.
# FINETUNED_CHECKPOINT_DIR=$MY_PROJECT_HOME/tmp/flowers-models/resnet_v1_50_titan/train
FINETUNED_CHECKPOINT_DIR=$MY_PROJECT_HOME/tmp/cub200-models/resnet_v1_50_glb_only/train_all/model.ckpt-40000

# Where the dataset is saved to.
DATASET_DIR=$MY_PROJECT_HOME/tmp/cub200/train_test_split

# input variables 
kept_percentage=0.4 
learning_rate=0.2 

# Where the training checkpoint and logs will be saved to.
TRAIN_DIR=$MY_PROJECT_HOME/tmp/cub200-models/prune/localtrain_manual_configs_lr${learning_rate}

# default option: use fixed learning rate=0.0001 (1e-4)
python p1_train_manual_configs.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=cub200 \
  --train_dataset_name=train \
  --test_dataset_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=resnet_v1_50 \
  --checkpoint_path=${FINETUNED_CHECKPOINT_DIR} \
  --max_number_of_steps=10000 \
  --batch_size=32 \
  --test_batch_size=100 \
  --learning_rate=${learning_rate} \
  --learning_rate_decay_type=fixed \
  --log_every_n_steps=100 \
  --summary_every_n_steps=500 \
  --evaluate_every_n_steps=500 \
  --runmeta_every_n_steps=13600 \
  --optimizer=sgd \
  --weight_decay=0.0001 \
  --kept_percentages=${kept_percentage} \
  --max_to_keep=1 \
  --continue_training=True

