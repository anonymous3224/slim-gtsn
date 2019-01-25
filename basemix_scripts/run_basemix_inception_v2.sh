#!/bin/bash
# This script globally train a pruned network based on a kept_percentage value 
set -e
# Where the fine-tuned ResNetV1-50 checkpoint is saved to.
FINETUNED_CHECKPOINT_DIR=$MY_PROJECT_HOME/tmp/cub200-models/inception_v2_glb_only/train_all/model.ckpt-40000/model.ckpt-40000

# Where the dataset is saved to.
DATASET_DIR=$MY_PROJECT_HOME/tmp/cub200/train_test_split

# dataset name 
DATASET_NAME=cub200 

# input variables
start_config_id=0 #$1
num_configs=1 #$2
learning_rate=0.001

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=$MY_PROJECT_HOME/tmp/cub200-models/prune_inception_v2/basemix_lr${learning_rate}

python basemix_inception_v2.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET_NAME} \
  --train_dataset_name=train \
  --test_dataset_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v2 \
  --checkpoint_path=${FINETUNED_CHECKPOINT_DIR} \
  --max_number_of_steps=30000 \
  --batch_size=32 \
  --test_batch_size=100 \
  --learning_rate=${learning_rate} \
  --learning_rate_decay_type=fixed \
  --log_every_n_steps=100 \
  --summary_every_n_steps=1000 \
  --evaluate_every_n_steps=500 \
  --runmeta_every_n_steps=55500 \
  --optimizer=sgd \
  --weight_decay=0.00001 \
  --start_configuration_index=${start_config_id} \
  --num_configurations=${num_configs}   \
  --continue_training=False \
  --max_to_keep=1

 


