# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



# Modified by Hui Guan
# phase 1: concurrent local training


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.client import timeline

from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
import numpy as np
import re, time,  math , os, sys, itertools 
from datetime import datetime
from pprint import pprint

from train_helper import *
from train_helper_for_vgg_16 import * 

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'num_readers', 10,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 16,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'summary_every_n_steps', 50,
    'The frequency with which summary op are done.')

tf.app.flags.DEFINE_integer(
    'evaluate_every_n_steps', 100,
    'The frequency with which evaluation are done.')

tf.app.flags.DEFINE_integer(
    'runmeta_every_n_steps', 1000,
    'The frequency with which RunMetadata are done.')



tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'train_image_size', 224, 'Train image size')

tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

#===============================
## added by Hui Guan, Oct. 17th 
#===============================
tf.app.flags.DEFINE_integer(
    'block_size', 2,
    'The number of convolutional layers inside a block. The mininum value is 2. ')

# tf.app.flags.DEFINE_integer(
#     'block_id', 0,
#     'The block to be pruned and trained. Index starts with 0.')

tf.app.flags.DEFINE_integer(
    'block_config_id', 0,
    'The configuration id for the block.')

# tf.app.flags.DEFINE_string(
#     'net_name_scope', 'resnet_v1_50',
#     'The name scope of previous trained network in the current graph.')

tf.app.flags.DEFINE_string(
    'net_name_scope_pruned', 'vgg_16_pruned',
    'The name scope for the pruned network in the current graph')

tf.app.flags.DEFINE_string(
    'net_name_scope_checkpoint', 'vgg_16',
    'The name scope for the saved previous trained network')

tf.app.flags.DEFINE_string(
    'kept_percentages', '0.3,0.5,0.7', 'The number of filters to be kept in a conv layer.')

tf.app.flags.DEFINE_integer(
    'test_batch_size', 32, 'The number of samples in each batch for test dataset.')

tf.app.flags.DEFINE_string(
    'train_dataset_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'test_dataset_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_integer(
    'max_to_keep', 5, 'The number of models to keep.')

tf.app.flags.DEFINE_boolean(
    'continue_training', False,
    'if continue training is true, then do not clean the train directory.')

tf.app.flags.DEFINE_boolean(
    'last_conv_pruned', False,
    'if true, the last convolutional layer in a block is pruned.')

FLAGS = tf.app.flags.FLAGS

def get_init_values_for_pruned_layers(prune_scopes, shorten_scopes, kept_percentage):
    """ prune layers iteratively so prune_scopes and shorten scopes should be of size one. """
    graph = tf.Graph()
    with graph.as_default():
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=FLAGS.num_clones,
            clone_on_cpu=FLAGS.clone_on_cpu,
            replica_id=FLAGS.task,
            num_replicas=FLAGS.worker_replicas,
            num_ps_tasks=FLAGS.num_ps_tasks)
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, 'train', FLAGS.dataset_dir)
        batch_queue = train_inputs(dataset, deploy_config, FLAGS)
        images, _ = batch_queue.dequeue()


        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            weight_decay=FLAGS.weight_decay
            )
        network_fn(images, is_training=False)
        
        sess = tf.Session()
        load_checkpoint(sess, FLAGS.checkpoint_path)

        print('HG: calculate variables init value for pruned network ...')
        variables_init_value, pruned_filter_indexes = get_pruned_kernel_matrix(sess, prune_scopes, shorten_scopes, kept_percentage)
    return variables_init_value, pruned_filter_indexes 


def main(_):
    tic = time.time()
    print('tensorflow version:', tf.__version__)
    tf.logging.set_verbosity(tf.logging.INFO)
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    # initialize constants
    net_name_scope_pruned = FLAGS.net_name_scope_pruned
    net_name_scope_checkpoint = FLAGS.net_name_scope_checkpoint
    kp_options = sorted([float(x) for x in FLAGS.kept_percentages.split(',')])
    num_blocks = int(len(valid_layer_names)/FLAGS.block_size)
    if len(valid_layer_names)%FLAGS.block_size!=0:
        print('ERROR: len(valid_layer_names)%FLAGS.block_size!=0')
        return 
    
    # if FLAGS.block_id >= num_blocks:
    #     print('ERROR: block_id=%d should be smaller than the number of blocks=%d' %(FLAGS.block_id, num_blocks))
    #     return 
    print('HG: kp_options', kp_options)
    print('HG: block size:', FLAGS.block_size)
    print('HG: number of blocks:', num_blocks) #, ', block_id:', FLAGS.block_id)
    if len(kp_options)>1:
        print('ERROR: only support kp options = 1')
        return 

    # prepare file system 
    # block_config_str = '_'.join(map(str, block_config))
    if FLAGS.last_conv_pruned:
        foldername = 'last_conv_pruned'
    else:
        foldername = 'last_conv_unpruned'
    results_dir = os.path.join(FLAGS.train_dir, foldername, 'kp'+str(FLAGS.kept_percentages)) 
    train_dir = os.path.join(results_dir, 'train')
    print('HG: train_dir:', train_dir)
    if not (FLAGS.continue_training and tf.train.latest_checkpoint(train_dir)):
        prepare_file_system(train_dir)


    def write_log_info(info):
        with open(os.path.join(FLAGS.train_dir, 'log.txt'), 'a') as f:
            f.write(info+'\n')
    def write_detailed_info(info):
        with open(os.path.join(train_dir, 'train_details.txt'), 'a') as f:
            f.write(info+'\n')

    info = 'train_dir:'+train_dir+'\n'
    info += 'kp_options:'+str(kp_options)+'\n'
    log_info = info+'\n'
    write_detailed_info(info)

    # set prune info 
    prune_info = {}
    block_layer_names_list = [] 
    pruned_layer_names_list = [] 
    block_config_list = [] 


    prune_scopes = []
    shorten_scopes = []  
    network_config = [] 

    for block_id in xrange(num_blocks):
        #------------------------
        # get block layer names 
        #------------------------
        start_layer_id = FLAGS.block_size*block_id
        end_layer_id = start_layer_id+FLAGS.block_size
        block_layer_names = valid_layer_names[start_layer_id:end_layer_id]
        is_last_block = valid_layer_names[-1] in block_layer_names
        # print('HG: block_layer_names:', block_layer_names)
        # print('HG: is_last_block:', is_last_block)
        block_layer_names_list.append(block_layer_names)

        #------------------------
        # get pruned layer names so that the pruned block can fit into the test network. 
        #------------------------
        pruned_layer_names = valid_layer_names[start_layer_id:end_layer_id]  
        config_length = FLAGS.block_size 
        # note that last layer cannot be pruned. 
        if is_last_block:
            pruned_layer_names.remove(valid_layer_names[-1])
            config_length -=1
    
        # if the block is not the first block, prune also the layer before the block 
        if block_id !=0:
            pruned_layer_names = [valid_layer_names[start_layer_id-1]] + pruned_layer_names 
            config_length +=1    
        # print('HG: pruned_layer_names:', pruned_layer_names)
        pruned_layer_names_list.append(pruned_layer_names)

        #------------------------
        # get the pruning configuration for the block 
        #------------------------
        # given N=#options, m=block_size. first block has #configs=N^m, other blocks have #configs=N^{m+1}, 
        block_configurations = list(itertools.product(kp_options, repeat=config_length))
        # print('HG: number of block variants:', len(block_configurations))
        if FLAGS.block_config_id >= len(block_configurations):
            print('ERROR: block_config_id=%d should be smaller than number of block variants=%d' %(FLAGS.block_config_id, len(block_configurations)))
            return 
        block_config = list(block_configurations[FLAGS.block_config_id])
        if not FLAGS.last_conv_pruned:
            # fix the input layer to the block to be 1.0
            if block_id !=0:
                block_config[0] = 1.0
            # fix the last conv layer in the block to be 1.0
            if not is_last_block:
                block_config[-1] = 1.0
        # block_config=[0.5, 0.5, 0.5]
        # print('HG: block confiugrations:', block_config)
        block_config_list.append(block_config)

        #------------------------
        # prepare prune_info with the config
        #------------------------
        for i in xrange(len(block_config)):
            layer_name = pruned_layer_names[i]
            if layer_name in prune_info:
                if prune_info[layer_name]['kp'] != block_config[i]:
                    print ("ERROR: layer name already in prune info but prune_info[layer_name]['kp'] != block_config[i]")
                    return 
            prune_info[layer_name]={'kp': block_config[i]}

        #------------------------
        # get pruned scopes and shorten scopes 
        #------------------------
        for layer_name, layer_config in zip(pruned_layer_names, block_config):
            prune_scope = layer_name_to_prune_scope(layer_name, net_name_scope_checkpoint)
            if prune_scope not in prune_scopes:
                prune_scopes.append(prune_scope)
                shorten_scopes.append(layer_name_to_shorten_scope(layer_name, net_name_scope_checkpoint))
                network_config.append(layer_config)

    #------------------------
    # get variables init value and pruned filter indexes 
    #------------------------
    print('HG: prune_scopes', len(prune_scopes) ,prune_scopes)
    print('HG: network_config', network_config)
    variables_init_value, pruned_filter_indexes = get_init_values_for_pruned_layers(prune_scopes, shorten_scopes, network_config)


    # print('HG: prune_info')
    # pprint(prune_info)

   
    with tf.Graph().as_default():
        #######################
        # Config model_deploy #
        #######################
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=FLAGS.num_clones,
            clone_on_cpu=FLAGS.clone_on_cpu,
            replica_id=FLAGS.task,
            num_replicas=FLAGS.worker_replicas,
            num_ps_tasks=FLAGS.num_ps_tasks)


        ######################
        # Select the dataset #
        ######################
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.train_dataset_name, FLAGS.dataset_dir)
        test_dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.test_dataset_name, FLAGS.dataset_dir)

        batch_queue = train_inputs(dataset, deploy_config, FLAGS)
        test_images, test_labels = test_inputs(test_dataset, deploy_config, FLAGS)
        images, labels = batch_queue.dequeue()

        ######################
        # Select the network#
        ######################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            weight_decay=FLAGS.weight_decay
            )
        _, end_points = network_fn(images, is_training=False)
        # for item in end_points.iteritems():
        #     print(item)
        # return

        network_fn_pruned = nets_factory.get_network_fn_pruned(
            FLAGS.model_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            weight_decay=FLAGS.weight_decay)

        # #########################################
        # # Configure the optimization procedure. #
        # #########################################
        with tf.device(deploy_config.variables_device()):
            global_step = tf.Variable(0, trainable=False, name='global_step')
        with tf.device(deploy_config.optimizer_device()):
            learning_rate = configure_learning_rate(dataset.num_samples, global_step, FLAGS)
            optimizer = configure_optimizer(learning_rate, FLAGS)
            tf.summary.scalar('learning_rate', learning_rate)


        ####################
        # Define the model #
        ####################
        # prune_info = {layer_name_1: {'kp': 0.3, 'inputs': inputs}, layer_name_2:{'kp':0.5}}
        # checkpoint_prune_info = {pruned_layer_names[-1]:{'kp':block_config[-1]}}
        # _, end_points = network_fn_pruned(images, 
        #                                 prune_info = checkpoint_prune_info, 
        #                                 is_training=True, 
        #                                 is_local_train=False, 
        #                                 reuse_variables=False,
        #                                 scope = net_name_scope_checkpoint)
        # --------------------------
        # set inputs for prune_info
        # --------------------------
        print_list('pruned_layer_names_list', pruned_layer_names_list)
        for block_id in xrange(num_blocks):
            block_layer_names = block_layer_names_list[block_id] 
            pruned_layer_names = pruned_layer_names_list[block_id]

            if block_id ==0:
                block_inputs = images
            else:
                # original inputs might have a different dimension with the required inputs dimension. 
                # use pruned_filter_indexes to prune original_inputs
                with tf.name_scope('inputs_selector_'+str(block_id)):
                    # get pruned filter indexes 
                    
                    prune_scope = net_name_scope_checkpoint+'/'+ pruned_layer_names[0]
                    filter_indexes = pruned_filter_indexes[prune_scope]

                    # get original inputs 
                    inputs_layer_id = all_layer_names.index(block_layer_names[0])-1
                    inputs_scope = net_name_scope_checkpoint+'/'+all_layer_names[inputs_layer_id]
                    original_inputs = end_points[inputs_scope]

                    # downsample original inputs 
                    num_dim = original_inputs.get_shape().as_list()[-1]
                    block_inputs = tf.stack([original_inputs[:,:,:,i] for i in xrange(num_dim) if i not in filter_indexes], axis=-1)
                    print('HG: inputs_scope:', inputs_scope, original_inputs.get_shape().as_list(), ' to ', block_inputs.get_shape().as_list())
            # set inputs for this block 
            prune_info[block_layer_names[0]]['inputs'] = block_inputs

        print('HG: prune_info:')
        pprint(prune_info)

        # generate the pruned network for training
        _, end_points_pruned = network_fn_pruned(images, 
                                                prune_info = prune_info, 
                                                is_training=True, 
                                                is_local_train=True, 
                                                reuse_variables=False,
                                                scope = net_name_scope_pruned)
        # generate the pruned network for testing
        logits, _ = network_fn_pruned(test_images, 
        							  prune_info = prune_info, 
        							  is_training=False, 
        							  is_local_train=False, 
        							  reuse_variables=True,
        							  scope = net_name_scope_pruned)
        # add correct prediction to the testing network 
        correct_prediction = add_correct_prediction(logits, test_labels)

        #############################
        # Specify the loss functions #
        #############################
        total_losses = [] 
        train_tensors = [] 
        for block_id in xrange(num_blocks):
            print('\n\nblock_id', block_id)
            block_layer_names = block_layer_names_list[block_id] 
            pruned_layer_names = pruned_layer_names_list[block_id]
            is_last_block = (block_id == num_blocks-1)

            outputs_scope = net_name_scope_checkpoint+'/'+block_layer_names[-1]
            outputs_scope_pruned = net_name_scope_pruned+'/'+block_layer_names[-1]
            print('HG: outputs_scope:', outputs_scope)
        
        
            # add reconstruction loss 
            collection_name = 'subgraph_losses_'+str(block_id)
            if outputs_scope not in end_points:
                raise ValueError('end_points does not contain the outputs_scope: %s', outputs_scope)
            outputs = end_points[outputs_scope]

            if outputs_scope_pruned not in end_points_pruned:
                raise ValueError('end_points_pruned does not contain the outputs_scope_pruned: %s', outputs_scope_pruned)
            outputs_pruned = end_points_pruned[outputs_scope_pruned]

            # TODO: cannot use l2_loss directory since the outputs and outputs_pruned do not have the same dimension.
            if is_last_block:
                outputs_gnd = outputs 
            else:
                with tf.name_scope('output_selector_'+str(block_id)):
                    filter_indexes = pruned_filter_indexes[outputs_scope]
                    num_dim = outputs.get_shape().as_list()[-1]
                    outputs_gnd =  tf.stack([outputs[:,:,:,i] for i in xrange(num_dim) if i not in filter_indexes], axis=-1)
                    print('HG: ouputs selector:', outputs.get_shape().as_list(), ' to ', outputs_gnd.get_shape().as_list())
            l2_loss = add_l2_loss(outputs_gnd, outputs_pruned, add_to_collection=True, collection_name=collection_name) 
              

            # get regularization loss
            train_scopes = [net_name_scope_pruned+'/'+item for item in block_layer_names]
            print_list('train_scopes', train_scopes)
            regularization_losses = get_regularization_losses_within_scopes(train_scopes, add_to_collection=True, collection_name=collection_name)
            print_list('regularization_losses', regularization_losses)

            # total loss and its summary
            total_loss = tf.add_n(tf.get_collection(collection_name), name='total_loss')
            for l in tf.get_collection(collection_name)+[total_loss]:
                tf.summary.scalar(l.op.name+'/summary', l)
            total_losses.append(total_loss)


            #############################
            # Add train operation       #
            #############################
            
            variables_to_train = get_trainable_variables_within_scopes(train_scopes)    
            print_list("variables_to_train", variables_to_train)    
            train_op = add_train_op(optimizer, total_loss, global_step, var_list=variables_to_train)
            with tf.control_dependencies([train_op]):
                train_tensor = tf.identity(total_loss, name='train_op')
            train_tensors.append(train_tensor)

        # add summary op
        summary_op = tf.summary.merge_all()

    
        print("HG: trainable_variables=", len(tf.trainable_variables()))
        print("HG: model_variables=", len(tf.model_variables()))
        print("HG: global_variables=", len(tf.global_variables()))
        print_list('model_variables but not trainable variables', list(set(tf.model_variables()).difference(tf.trainable_variables())))
        print_list('global_variables but not model variables', list(set(tf.global_variables()).difference(tf.model_variables())))
        print("HG: trainable_variables from "+net_name_scope_checkpoint+"=", len(get_trainable_variables_within_scopes([net_name_scope_checkpoint+'/'])))
        print("HG: trainable_variables from "+net_name_scope_pruned+"=", len(get_trainable_variables_within_scopes([net_name_scope_pruned+'/'])))
        print("HG: model_variables from "+net_name_scope_checkpoint+"=", len(get_model_variables_within_scopes([net_name_scope_checkpoint+'/'])))
        print("HG: model_variables from "+net_name_scope_pruned+"=", len(get_model_variables_within_scopes([net_name_scope_pruned+'/'])))
        print("HG: global_variables from "+net_name_scope_checkpoint+"=", len(get_global_variables_within_scopes([net_name_scope_checkpoint+'/'])))
        print("HG: global_variables from "+net_name_scope_pruned+"=", len(get_global_variables_within_scopes([net_name_scope_pruned+'/'])))
        
        sess_config = tf.ConfigProto(intra_op_parallelism_threads=16,
                                        inter_op_parallelism_threads=16)

        with tf.Session(config=sess_config) as sess:
            ###########################
            # prepare for filewritter #
            ###########################
            train_writer = tf.summary.FileWriter(train_dir, sess.graph)
            
            if not (FLAGS.continue_training and tf.train.latest_checkpoint(train_dir)):
                ###########################################
                # Restore original model variable values. #
                ###########################################
                
                variables_to_restore = get_model_variables_within_scopes([net_name_scope_checkpoint+'/'])
                print_list("restore model variables for original", variables_to_restore)
                load_checkpoint(sess, FLAGS.checkpoint_path, var_list=variables_to_restore)

                #################################################
                # Init  pruned networks  with  well-trained model #
                #################################################
                variables_to_reinit = get_model_variables_within_scopes([net_name_scope_pruned+'/'])
                print_list("init pruned model variables for pruned network", variables_to_reinit)
                assign_ops = []
                for v in variables_to_reinit:
                    key = re.sub(net_name_scope_pruned, net_name_scope_checkpoint, v.op.name)
                    if key in variables_init_value:
                        value = variables_init_value.get(key)
                        # print(key, value) 
                        assign_ops.append(tf.assign(v, tf.convert_to_tensor(value), validate_shape=True))
                        # v.set_shape(value.shape)
                    else:
                        raise ValueError("Key not in variables_init_value, key=", key)
                assign_op = tf.group(*assign_ops)
                sess.run(assign_op)

                
            else:
                # restore all variables from checkpoint
                variables_to_restore = get_global_variables_within_scopes()
                load_checkpoint(sess, train_dir, var_list=variables_to_restore)

            #################################################
            # init unitialized global variable. #
            #################################################
            
            variables_to_init = get_global_variables_within_scopes(sess.run( tf.report_uninitialized_variables() ))
            print_list("init uninitialized_variables", variables_to_init)
            sess.run( tf.variables_initializer(variables_to_init) )
            
            init_global_step_value = sess.run(global_step)
            print('initial global step: ', init_global_step_value)
            if init_global_step_value >= FLAGS.max_number_of_steps:
                print('Exit: init_global_step_value (%d) >= FLAGS.max_number_of_steps (%d)' \
                    %(init_global_step_value, FLAGS.max_number_of_steps))
                return

            ###########################
            # Record CPU usage  #
            ###########################
            # mpstat_output_filename = os.path.join(train_dir, "cpu-usage.log")
            # os.system("mpstat -P ALL 1 > " + mpstat_output_filename + " 2>&1 &")

            ###########################
            # Kicks off the training. #
            ###########################
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            print('HG: # of threads=', len(threads))

            # saver for models 
            if FLAGS.max_to_keep<=0:
                max_to_keep =int(2*FLAGS.max_number_of_steps/FLAGS.evaluate_every_n_steps)
            else:
                max_to_keep = FLAGS.max_to_keep 
            saver = tf.train.Saver(max_to_keep=max_to_keep)

            train_time = 0 # the amount of time spending on sgd training only. 
            duration = 0 # used to estimate the training speed
            train_only_cnt = 0  # used to calculate the true training time. 
            duration_cnt = 0 

            print("start to train at:", datetime.now())
            for i in range(init_global_step_value, FLAGS.max_number_of_steps+1):

                # run optional meta data, or summary, while run train tensor
                if i > init_global_step_value: # FLAGS.max_number_of_steps:
                    # run metadata
                    if i % FLAGS.runmeta_every_n_steps == FLAGS.runmeta_every_n_steps-1:
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()

                        loss_values = sess.run(train_tensors,
                                              options = run_options,
                                              run_metadata=run_metadata)
                        train_writer.add_run_metadata(run_metadata, 'step%d-train' % i)

                        # Create the Timeline object, and write it to a json file
                        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                        chrome_trace = fetched_timeline.generate_chrome_trace_format()
                        with open(os.path.join(train_dir, 'timeline_'+str(i)+'.json'), 'w') as f:
                            f.write(chrome_trace)

                    # record summary
                    elif i % FLAGS.summary_every_n_steps==0:
                        results = sess.run([summary_op]+ train_tensors)
                        train_summary, loss_values= results[0], results[-1]
                        train_writer.add_summary(train_summary, i)
                        # print('HG: train with summary')
                        # only run train op
                    else:
                        start_time = time.time()
                        loss_values = sess.run(train_tensors)
                        train_only_cnt+=1
                        duration_cnt +=1 
                        train_time += time.time() - start_time 
                        duration += time.time()- start_time 

                    if i%FLAGS.log_every_n_steps==0 and duration_cnt > 0:
                        # record speed
                        log_frequency = duration_cnt
                        examples_per_sec = log_frequency * FLAGS.batch_size / duration
                        sec_per_batch = float(duration /log_frequency)
                        summary = tf.Summary()
                        summary.value.add(tag='examples_per_sec', simple_value=examples_per_sec)
                        summary.value.add(tag='sec_per_batch', simple_value=sec_per_batch)
                        train_writer.add_summary(summary, i)
                        info = ('%s: step %d, loss = %s (%.1f examples/sec; %.3f sec/batch)') % (datetime.now(), i, str(loss_values), examples_per_sec, sec_per_batch)
                        print(info)
                        duration = 0 
                        duration_cnt = 0 

                        write_detailed_info(info)
                else: 
                    # run only total loss when i=0 
                    results = sess.run([summary_op]+total_losses) #loss_value = sess.run(total_loss)
                    train_summary, loss_values = results[0], results[-1]
                    train_writer.add_summary(train_summary, i)
                    format_str = ('%s: step %d, loss = %s')
                    print(format_str % (datetime.now(), i, str(loss_values)))
                    info= format_str % (datetime.now(), i, str(loss_values))
                    write_detailed_info(info)

                # record the evaluation accuracy
                is_last_step = (i==FLAGS.max_number_of_steps)
                if i%FLAGS.evaluate_every_n_steps==0 or is_last_step:

                    test_accuracy, run_metadata = evaluate_accuracy(sess, coord, test_dataset.num_samples,
                                  test_images, test_labels, test_images, test_labels, 
                                  correct_prediction, FLAGS.test_batch_size, run_meta=False)
                    summary = tf.Summary()
                    summary.value.add(tag='accuracy', simple_value=test_accuracy)
                    train_writer.add_summary(summary, i)
                    # if run_meta: 
                        # eval_writer.add_run_metadata(run_metadata, 'step%d-eval' % i)
                    info = ('%s: step %d, test_accuracy = %s') % (datetime.now(), i,  str(test_accuracy))
                    print(info)
                    if i==init_global_step_value or is_last_step:
                        # write_log_info(info)
                        log_info += info +'\n'
                    write_detailed_info(info)

                    ###########################
                    # Save model parameters . #
                    ###########################
                    # saver = tf.train.Saver()
                    save_path = saver.save(sess, os.path.join(train_dir, 'model.ckpt-'+str(i)))
                    print("HG: Model saved in file: %s" % save_path)

            coord.request_stop()
            coord.join(threads)
            total_time = time.time()-tic 

            train_speed = train_time*1.0/train_only_cnt
            train_time = train_speed*(FLAGS.max_number_of_steps) 
            info = "HG: training speed(sec/batch): %.6f\n" %(train_speed)
            info += "HG: training time(min): %.1f, total time(min): %.1f" %( train_time/60.0,  total_time/60.0)
            print(info)
            log_info+=info+'\n\n'
            write_log_info(log_info)
            write_detailed_info(info)
if __name__ == '__main__':
    tf.app.run()

