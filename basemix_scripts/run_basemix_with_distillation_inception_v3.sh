#!/bin/bash
# This script globally train a pruned network based on a kept_percentage value 
set -e
# Where the fine-tuned ResNetV1-50 checkpoint is saved to.
FINETUNED_CHECKPOINT_DIR=$MY_PROJECT_HOME/tmp/cub200-models/inception_v3_glb_only/train_all/model.ckpt-40000

# Where the dataset is saved to.
DATASET_DIR=$MY_PROJECT_HOME/tmp/cub200/train_test_split

# input variables 
learning_rate=0.005

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=$MY_PROJECT_HOME/tmp/cub200-models/prune_inception_v3/entropy_logits_lr${learning_rate}


# input variables
for start_config_id in 0
do 

python basemix_with_distillation_inception_v3.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=cub200 \
  --train_dataset_name=train \
  --test_dataset_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3 \
  --checkpoint_path=${FINETUNED_CHECKPOINT_DIR} \
  --max_number_of_steps=30000 \
  --batch_size=32 \
  --test_batch_size=100 \
  --learning_rate=${learning_rate} \
  --learning_rate_decay_type=fixed \
  --log_every_n_steps=100 \
  --summary_every_n_steps=500 \
  --evaluate_every_n_steps=1000 \
  --runmeta_every_n_steps=55500 \
  --optimizer=sgd \
  --weight_decay=0.00001 \
  --start_configuration_index=${start_config_id} \
  --total_num_configurations=500 \
  --logits_loss_lambda=1 \
  --include_cross_entropy=True \
  --continue_training=False \
  --max_to_keep=1

done
 


