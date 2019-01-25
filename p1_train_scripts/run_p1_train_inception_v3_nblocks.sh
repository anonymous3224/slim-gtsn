#!/bin/bash
#
# This script performs the following operations:
# locally train pruned blocks on the computer:
echo $COMPUTER_NAME
set -e

# Where the fine-tuned ResNetV1-50 checkpoint is saved to.
FINETUNED_CHECKPOINT_DIR=$MY_PROJECT_HOME/tmp/cub200-models/inception_v3_glb_only/train_all/model.ckpt-40000

# Where the dataset is saved to.
DATASET_DIR=$MY_PROJECT_HOME/tmp/cub200/train_test_split

# input variables 
# kept_percentage=$1 
# learning_rate=$2

for kept_percentage in 0.3 #0.5 0.7
do 
  for block_size in 4 #4 8 16 
  do 
learning_rate=0.08

# Where the training checkpoint and logs will be saved to.
TRAIN_DIR=$MY_PROJECT_HOME/tmp/cub200-models/prune_inception_v3/exp3/localtrain_lr${learning_rate}_bs${block_size}

# default option: use fixed learning rate=0.0001 (1e-4)
python p1_train_inception_v3_nblocks.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=cub200 \
  --train_dataset_name=train \
  --test_dataset_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3 \
  --checkpoint_path=${FINETUNED_CHECKPOINT_DIR} \
  --max_number_of_steps=20000 \
  --batch_size=32 \
  --test_batch_size=100 \
  --learning_rate=${learning_rate} \
  --learning_rate_decay_type=fixed \
  --log_every_n_steps=100 \
  --summary_every_n_steps=500 \
  --evaluate_every_n_steps=500 \
  --runmeta_every_n_steps=136000 \
  --optimizer=sgd \
  --weight_decay=0.0001 \
  --kept_percentages=${kept_percentage} \
  --block_size=${block_size} \
  --max_to_keep=1 \
  --continue_training=True

done 
done 

