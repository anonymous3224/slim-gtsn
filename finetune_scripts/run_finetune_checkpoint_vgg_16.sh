#!/bin/bash
# this script finetune a trained model to a new dataset. 

set -e
# Where the fine-tuned ResNetV1-50 checkpoint is saved to.
FINETUNED_CHECKPOINT_DIR=$MY_PROJECT_HOME/tmp/checkpoints/vgg_16.ckpt 

# Where the dataset is saved to.
# DATASET_DIR=$MY_PROJECT_HOME/tmp/flowers
DATASET_DIR=$MY_PROJECT_HOME/tmp/cub200/train_test_split

# the dataset to use 
DATASET_NAME=cub200

# Where the training (fine-tuned) checkpoint and logs will be saved to.
LOG_DIR=$MY_PROJECT_HOME/tmp/cub200-models/vgg_16_glb_only


TRAIN_ALL_DIR=${LOG_DIR}/train_all_dcy0.0005
#EVAL_ALL_DIR=${LOG_DIR}/eval_all

 
# train option 2: directly train entire network 
python finetune_checkpoint.py \
  --train_dir=${TRAIN_ALL_DIR} \
  --dataset_name=${DATASET_NAME} \
  --train_dataset_name=train \
  --test_dataset_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=vgg_16 \
  --checkpoint_path=${FINETUNED_CHECKPOINT_DIR} \
  --checkpoint_exclude_scopes=vgg_16/fc8 \
  --max_number_of_steps=40000 \
  --batch_size=32 \
  --test_batch_size=100 \
  --learning_rate=0.001 \
  --learning_rate_decay_type=fixed \
  --learning_rate_decay_factor=0.5 \
  --num_epochs_per_decay=30 \
  --log_every_n_steps=100 \
  --summary_every_n_steps=500 \
  --evaluate_every_n_steps=500 \
  --runmeta_every_n_steps=66600 \
  --optimizer=sgd \
  --weight_decay=0.0005 \
  --continue_training=True 

