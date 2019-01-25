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



# Modified by Hui Guan, since Sept. 27th, 2017
# prune once and train 
# prune all the valid layers and train the network with the objective being cross-entropy+regularization. 


# about batch normalization: http://ruishu.io/2016/12/27/batchnorm/

# ==============================================================================
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import mpi module, must be first import
# from mpi4py import MPI

import tensorflow as tf
from tensorflow.python.client import timeline

from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
import numpy as np
import re, time,  math , os, sys 
from datetime import datetime
from pprint import pprint 

from train_helper import *
from train_helper_for_resnet_v1_50 import * 

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

## added by Hui Guan, Oct. 17th 
# tf.app.flags.DEFINE_string(
#     'net_name_scope', 'resnet_v1_50',
#     'Previous trained network scope.')

tf.app.flags.DEFINE_string(
    'net_name_scope_checkpoint', 'resnet_v1_50',
    'The name scope for the saved previous trained network')

tf.app.flags.DEFINE_string(
    'net_name_scope_pruned', 'resnet_v1_50_pruned',
    'The name scope of pruned network in the current graph.')

tf.app.flags.DEFINE_float(
    'kept_percentage', 0.5,
    'The numbers of filters to keep')

tf.app.flags.DEFINE_integer(
    'block_id', 0,
    'The block to be pruned for sensitivity testing')


tf.app.flags.DEFINE_integer(
    'test_batch_size', 32, 'The number of samples in each batch for the test dataset.')

tf.app.flags.DEFINE_string(
    'train_dataset_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'test_dataset_name', 'val', 'The name of the train/test split.')

tf.app.flags.DEFINE_boolean(
    'continue_training', False,
    'if continue training is true, then do not clean the train directory.')

tf.app.flags.DEFINE_integer(
    'max_to_keep', 5, 'The number of models to keep.')

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

        print("HG: trainable_variables=", len(tf.trainable_variables()))
        print("HG: model_variables=", len(tf.model_variables()))
        print("HG: global_variables=", len(tf.global_variables()))
        
        with tf.Session() as sess:
            load_checkpoint(sess, FLAGS.checkpoint_path)
            variables_init_value = get_pruned_kernel_matrix(sess, prune_scopes, shorten_scopes, kept_percentage)
    # remove graph 
    del graph 
    return variables_init_value




def main(_):
    tic = time.time()
    print('tensorflow version:', tf.__version__)
    tf.logging.set_verbosity(tf.logging.INFO)
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    # init 
    net_name_scope_pruned = FLAGS.net_name_scope_pruned
    net_name_scope_checkpoint = FLAGS.net_name_scope_checkpoint
    indexed_prune_scopes_for_units = valid_indexed_prune_scopes_for_units
    kept_percentage = FLAGS.kept_percentage

    
    # set the configuration: should be a 16-length vector 
    config = [1.0]*len(indexed_prune_scopes_for_units)
    config[FLAGS.block_id] = kept_percentage
    print("config:", config)


    # prepare for training with the specific config 
    indexed_prune_scopes = indexed_prune_scopes_for_units[FLAGS.block_id]
    prune_info = indexed_prune_scopes_to_prune_info(indexed_prune_scopes, kept_percentage)
    print("prune_info:", prune_info )
    
    # prepare file system 
    results_dir = os.path.join(FLAGS.train_dir, 'id'+str(FLAGS.block_id)) #+'_'+str(FLAGS.max_number_of_steps))
    train_dir = os.path.join(results_dir, 'kp'+str(kept_percentage))


    prune_scopes = indexed_prune_scopes_to_prune_scopes(indexed_prune_scopes, net_name_scope_checkpoint)
    shorten_scopes = indexed_prune_scopes_to_shorten_scopes(indexed_prune_scopes, net_name_scope_checkpoint)
    variables_init_value = get_init_values_for_pruned_layers(prune_scopes, shorten_scopes, kept_percentage)
    reinit_scopes = [re.sub(net_name_scope_checkpoint, net_name_scope_pruned, v) for v in prune_scopes+shorten_scopes]
    
    prepare_file_system(train_dir)

    def write_detailed_info(info):
        with open(os.path.join(train_dir, 'eval_details.txt'), 'a') as f:
            f.write(info+'\n')
    info = 'train_dir:'+train_dir+'\n'
    info += 'block_id:'+ str(FLAGS.block_id)+'\n'
    info += 'configuration: '+ str(config)+'\n'
    info += 'indexed_prune_scopes: ' + str(indexed_prune_scopes)+'\n'
    info += 'kept_percentage: ' + str(kept_percentage)
    print(info)
    write_detailed_info(info)

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
        # dataset = dataset_factory.get_dataset(
        #     FLAGS.dataset_name, FLAGS.train_dataset_name, FLAGS.dataset_dir)
        test_dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.test_dataset_name , FLAGS.dataset_dir)

        # batch_queue = train_inputs(dataset, deploy_config, FLAGS)
        test_images, test_labels = test_inputs(test_dataset, deploy_config, FLAGS)
        # images, labels = batch_queue.dequeue()

        ######################
        # Select the network#
        ######################
        
        network_fn_pruned = nets_factory.get_network_fn_pruned(
            FLAGS.model_name,
            prune_info=prune_info,
            num_classes=(test_dataset.num_classes - FLAGS.labels_offset),
            weight_decay=FLAGS.weight_decay)
        print('HG: prune_info:')
        pprint(prune_info)

        ####################
        # Define the model #
        ####################
        # logits_train, _ = network_fn_pruned(images, is_training=True, is_local_train=False, reuse_variables=False, scope = net_name_scope_pruned)
        logits_eval, _ = network_fn_pruned(test_images, is_training=False, is_local_train=False, reuse_variables=False, scope = net_name_scope_pruned)
        correct_prediction = add_correct_prediction(logits_eval, test_labels)

    
        print("HG: trainable_variables=", len(tf.trainable_variables()))
        print("HG: model_variables=", len(tf.model_variables()))
        print("HG: global_variables=", len(tf.global_variables()))
        
        sess_config = tf.ConfigProto(intra_op_parallelism_threads=16,
                                        inter_op_parallelism_threads=16)
        with tf.Session(config=sess_config) as sess:
            ###########################
            # Prepare for filewriter. #
            ###########################
            # train_writer = tf.summary.FileWriter(train_dir, sess.graph)


            #########################################
            # Reinit  pruned model variable  #
            #########################################
            variables_to_reinit = get_model_variables_within_scopes(reinit_scopes)
            print_list("Initialize pruned variables", variables_to_reinit)
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

            #################################################
            # Restore unchanged model variable. #
            #################################################
            variables_to_restore = {re.sub(net_name_scope_pruned, net_name_scope_checkpoint, v.op.name):
                          v for v in get_model_variables_within_scopes()
                          if v not in variables_to_reinit}
            print_list("restore model variables", variables_to_restore.values())
            load_checkpoint(sess, FLAGS.checkpoint_path, var_list=variables_to_restore)

            
            ###########################
            # Kicks off the training. #
            ###########################
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            # saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)
            print('HG: # of threads=', len(threads))
            
            eval_time = -1* time.time() 
            test_accuracy, run_metadata = evaluate_accuracy(sess, coord, test_dataset.num_samples,
                          test_images, test_labels, test_images, test_labels, 
                          correct_prediction, FLAGS.test_batch_size, run_meta=False)
            eval_time += time.time() 

            info=('%s: test_accuracy = %.6f') % (datetime.now(), test_accuracy)
            print(info)
            write_detailed_info(info)


            coord.request_stop()
            coord.join(threads)
            total_time = time.time()-tic 
            
            info = "HG: training time(min): %.1f, total time(min): %.1f" %( eval_time/60.0,  total_time/60.0)
            print(info)
            write_detailed_info(info)
            
            

if __name__ == '__main__':
    tf.app.run()
    
