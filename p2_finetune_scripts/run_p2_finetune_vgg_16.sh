#!/bin/bash
# This script run the second phase fine tuning on composed networks with pretrained building blocks. 
#
set -e

# Where the dataset is saved to.
DATASET_DIR=$MY_PROJECT_HOME/tmp/cub200/train_test_split

# where the pretrained model is saved to 
CHECKPOINT_DIR=$MY_PROJECT_HOME/tmp/cub200-models/prune_vgg_16/localtrain_lr0.000001

# get input variables 
start_config_id=0
learning_rate=0.001


TRAIN_DIR=$MY_PROJECT_HOME/tmp/cub200-models/prune_vgg_16/finetune_lr${learning_rate}

python p2_finetune_vgg_16.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=cub200 \
  --train_dataset_name=train \
  --test_dataset_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=vgg_16 \
  --checkpoint_path=${CHECKPOINT_DIR} \
  --max_number_of_steps=20000 \
  --batch_size=32 \
  --test_batch_size=50 \
  --learning_rate=${learning_rate} \
  --learning_rate_decay_type=fixed \
  --log_every_n_steps=100 \
  --summary_every_n_steps=500 \
  --evaluate_every_n_steps=500 \
  --runmeta_every_n_steps=55500 \
  --optimizer=sgd \
  --weight_decay=0.00001 \
  --kept_percentages=0.5 \
  --total_num_configs=500 \
  --config_type=special \
  --start_config_id=${start_config_id} \
  --continue_training=True \
  --max_to_keep=1 \
  --last_conv_pruned=False 




  

