#!/bin/bash
# This script run the second phase fine tuning on composed networks with pretrained building blocks. 
#
set -e

# Where the dataset is saved to.
DATASET_DIR=$MY_PROJECT_HOME/tmp/cub200/train_test_split

# where the pretrained model is saved to 
CHECKPOINT_DIR=$MY_PROJECT_HOME/tmp/cub200-models/prune/localtrain_manual_configs_lr0.2


# Where the training checkpoint and logs will be saved to.

learning_rate=0.001
TRAIN_DIR=$MY_PROJECT_HOME/tmp/cub200-models/prune_$COMPUTER_NAME/exp4/finetune_manual_configs_lr${learning_rate}


for config_id in `seq 0 15`
do 
python p2_finetune_manual_configs.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=cub200 \
  --train_dataset_name=train \
  --test_dataset_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=resnet_v1_50 \
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
  --config_id=${config_id} \
  --continue_training=True \
  --max_to_keep=1 

done 


  

