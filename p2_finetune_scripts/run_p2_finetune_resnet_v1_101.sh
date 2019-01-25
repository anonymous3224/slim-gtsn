#!/bin/bash
# This script run the second phase fine tuning on composed networks with pretrained building blocks. 
set -e

# Where the dataset is saved to.
DATASET_DIR=$MY_PROJECT_HOME/tmp/cub200/train_test_split

# where the pretrained model is saved to 
CHECKPOINT_DIR=$MY_PROJECT_HOME/tmp/cub200-models/prune_resnet_v1_101/localtrain_lr0.2

# get input variables 
start_config_id=$1
num_configs=$2 # the number of configurations to evaluate

# Where the training checkpoint and logs will be saved to.
learning_rate=0.001
TRAIN_DIR=$MY_PROJECT_HOME/tmp/cub200-models/prune_${COMPUTER_NAME}_resnet_v1_101/exp2/finetune_lr${learning_rate}

python p2_finetune_resnet_v1_101.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=cub200 \
  --train_dataset_name=train \
  --test_dataset_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=resnet_v1_101 \
  --checkpoint_path=${CHECKPOINT_DIR} \
  --max_number_of_steps=20000 \
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
  --kept_percentages=0.3,0.5,0.7 \
  --num_configurations=${num_configs} \
  --total_num_configurations=500 \
  --configuration_type=sample \
  --start_configuration_index=${start_config_id} \
  --continue_training=True \
  --max_to_keep=1 

