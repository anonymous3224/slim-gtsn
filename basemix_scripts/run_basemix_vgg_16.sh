#!/bin/bash
# This script globally train a pruned network based on a kept_percentage value 
set -e
# Where the fine-tuned ResNetV1-50 checkpoint is saved to.
FINETUNED_CHECKPOINT_DIR=$MY_PROJECT_HOME/tmp/cub200-models/vgg_16_glb_only/train_all/model.ckpt-40000

# Where the dataset is saved to.
DATASET_DIR=$MY_PROJECT_HOME/tmp/cub200/train_test_split

# input variables
start_config_id=1
learning_rate=0.001 #0.001

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=$MY_PROJECT_HOME/tmp/cub200-models/prune_vgg_16/basemix_lr${learning_rate}

python basemix_vgg_16.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=cub200 \
  --train_dataset_name=train \
  --test_dataset_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=vgg_16 \
  --checkpoint_path=${FINETUNED_CHECKPOINT_DIR} \
  --max_number_of_steps=20000 \
  --batch_size=32 \
  --test_batch_size=50 \
  --learning_rate=${learning_rate} \
  --learning_rate_decay_type=fixed \
  --log_every_n_steps=100 \
  --summary_every_n_steps=500 \
  --evaluate_every_n_steps=1000 \
  --runmeta_every_n_steps=55500 \
  --optimizer=sgd \
  --weight_decay=0.00001 \
  --start_config_id=${start_config_id} \
  --total_num_configs=500 \
  --continue_training=True \
  --max_to_keep=1 \
  --last_conv_pruned=False 

 


