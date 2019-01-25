#!/bin/bash
# This script run the second phase fine tuning on composed networks with pretrained building blocks. 
#
set -e

# Where the dataset is saved to.
DATASET_DIR=$MY_PROJECT_HOME/tmp/cub200/train_test_split

# where the pretrained model is saved to 
block_size=2
CHECKPOINT_DIR=$MY_PROJECT_HOME/tmp/cub200-models/prune_inception_v3/localtrain_lr0.08_bs${block_size}

# get input variables 
start_config_id=$1
# num_configs=$2 # the number of configurations to evaluate

# Where the training checkpoint and logs will be saved to.

learning_rate=0.001
TRAIN_DIR=$MY_PROJECT_HOME/tmp/cub200-models/prune_inception_v3/finetune_lr${learning_rate}_bs${block_size}

python p2_finetune_inception_v3_nblocks.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=cub200 \
  --train_dataset_name=train \
  --test_dataset_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3 \
  --checkpoint_path=${CHECKPOINT_DIR} \
  --max_number_of_steps=30000 \
  --batch_size=32 \
  --test_batch_size=100 \
  --learning_rate=${learning_rate} \
  --learning_rate_decay_type=fixed \
  --log_every_n_steps=100 \
  --summary_every_n_steps=500 \
  --evaluate_every_n_steps=500 \
  --runmeta_every_n_steps=55500 \
  --optimizer=sgd \
  --weight_decay=0.00001 \
  --kept_percentages=0.3,0.5,0.7 \
  --total_num_configurations=500 \
  --configuration_type=special \
  --start_configuration_index=${start_config_id} \
  --continue_training=True \
  --max_to_keep=1 

#  --local_train_steps=${local_train_steps}


  

