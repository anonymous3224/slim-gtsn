# GTSN: Enable Block-Level Reuse for Fast Training Pruned CNN

## Dataset 
* Datasets are at [Google Drive](https://drive.google.com/drive/folders/1Z-IWCOoIu3vyN-BnVsd_rVyeUJ4LxaoB?usp=sharing): /tmp/[DATASET_NAME], DATASET_NAME: {flowers102, cub200, cars, dogs})
* Datasets are in tfrecord format. 

## Checkpoints
* Well-trained models and finetuned checkpoints are stored at [Google Drive](https://drive.google.com/drive/folders/1Z-IWCOoIu3vyN-BnVsd_rVyeUJ4LxaoB?usp=sharing): ./tmp/[DATASET_NAME]-models/[MODEL_NAME]-glb_only (MODEL_NAME: {inception_v2, inception_v3, resnet_v1_50, resnet_v1_101, vgg_16})
* Finetune code: finetune_checkpoint.py
* Example script: ```./finetune_scripts/run_finetune_checkpoint_[MODEL_NAME].sh``` 

## Compare with Greedy Method
In this first experiment, we conduct a head-to-head com-
parison to a recent incremental training method for CNN
pruning. 

###### sensitivity study:
* Code: block_sensitivity.py 
* Example script: ```./basemix_scripts/run_block_sensitivity.sh```

###### Greedy Method:
* Code: basemix_greedy.py
* Example script:```./basemix_scripts/run_basemix_greedy.sh ```

###### GTSN-based Method:
* Stage 1: p1_train_manual_configs.py;
* Example script:```./p1_train_scripts/run_p1_train_manual_configs.sh ```
* Stage 2: p2_finetune_manual_configs.py
* Example script:```./p2_train_scripts/run_p2_finetune_manual_configs.sh ```
* configurations are in ./configs_greedy/


## Compare with Enumerative Method
In this experiment, we compare GTSN-based training with
the enumerative method. The objective of the pruning is 
to find the smallest network that meets a predefined accuracy
target.

###### Enumerative Method:
* Code: basemix_[MODEL_NAME].py 
* Example script: ```./basemix_scripts/run_basemix_[MODEL_NAME].sh```
* configurations are in ./configs_enum

###### Enumerative Method with Knowledge Distillation:
* Code: basemix_with_distillation_[MODEL_NAME].py 
* Example script: ```./basemix_scripts/run_basemix_with_distillation_[MODEL_NAME].sh```

###### GTSN-based Method:
* Stage 1: p1_train_[MODEL_NAME].py;
* Example script: ```./p1_train_scripts/run_p1_train_[MODEL_NAME].sh ```
* Stage 2: p2_finetune_manual_configs.py
* Example script: ```./p2_train_scripts/run_p2_finetune_[MODEL_NAME].sh ```

## Notes
* Tested on tensorflow version 1.3.0, 1.7.0
* Required packages: tensorflow, mpi4py
