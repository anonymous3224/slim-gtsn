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
import re, time,  math , os, sys
from datetime import datetime
from pprint import pprint

from train_helper import *
from train_helper_for_inception_v3 import * 

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
tf.app.flags.DEFINE_string(
    'current_indexed_prune_scopes', 'all',
    'current to be pruned scope. can be only one convolutional layers')

# tf.app.flags.DEFINE_string(
#     'net_name_scope', 'InceptionV3',
#     'The name scope of previous trained network in the current graph.')

tf.app.flags.DEFINE_string(
    'net_name_scope_pruned', 'InceptionV3_pruned',
    'The name scope for the pruned network in the current graph')

tf.app.flags.DEFINE_string(
    'net_name_scope_checkpoint', 'InceptionV3',
    'The name scope for the saved previous trained network')

tf.app.flags.DEFINE_string(
    'kept_percentages', '0.5', 'The number of filters to be kept in a conv layer.')

tf.app.flags.DEFINE_integer(
    'test_batch_size', 32, 'The number of samples in each batch for test dataset.')

tf.app.flags.DEFINE_string(
    'train_dataset_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'test_dataset_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_integer(
    'max_to_keep', 5, 'The number of models to keep.')

tf.app.flags.DEFINE_integer(
    'block_size', 2, 'The number of building blocks as the basic block to train.')

tf.app.flags.DEFINE_boolean(
    'continue_training', False,
    'if continue training is true, then do not clean the train directory.')

FLAGS = tf.app.flags.FLAGS



def main(_):
    tic = time.time()
    print('tensorflow version:', tf.__version__)
    tf.logging.set_verbosity(tf.logging.INFO)
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    # init 
    print('HG: Train pruned blocks for all valid layers concurrently')
    block_names = valid_block_names
    net_name_scope_checkpoint = FLAGS.net_name_scope_checkpoint
    kept_percentages = sorted([float(x) for x in FLAGS.kept_percentages.split(',')])
    print_list('kept_percentages', kept_percentages)

    # prepare file system 
    results_dir = os.path.join(FLAGS.train_dir, 'kp'+FLAGS.kept_percentages)
    train_dir = os.path.join(results_dir, 'train')
    if (not FLAGS.continue_training) or (not tf.train.latest_checkpoint(train_dir)):
        print('Start a new training')
        prepare_file_system(train_dir)
    else:
        print('Continue training')


    def write_log_info(info):
        with open(os.path.join(FLAGS.train_dir, 'log.txt'), 'a') as f:
            f.write(info+'\n')
    def write_detailed_info(info):
        with open(os.path.join(train_dir, 'train_details.txt'), 'a') as f:
            f.write(info+'\n')

    info = 'train_dir:'+train_dir
    log_info = info+'\n'
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
        # each kept_percentage corresponds to a pruned network. 
        train_tensors = []
        total_losses = [] 
        pruned_net_name_scopes = []
        correct_predictions = []
        prune_infos = [] 
        for kept_percentage in kept_percentages:

            prune_info = kept_percentage_sequence_to_prune_info(kept_percentage, block_names)
            set_prune_info_inputs(prune_info, end_points, block_size=FLAGS.block_size)
            pprint(prune_info)
            # return 
            prune_infos.append(prune_info)

            #  the pruned network scope
            net_name_scope_pruned = FLAGS.net_name_scope_pruned+'_p'+str(kept_percentage)
            pruned_net_name_scopes.append(net_name_scope_pruned)

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
            correct_predictions.append(correct_prediction)

            #############################
            # Specify the loss functions #
            #############################
            print('HG: block_size', FLAGS.block_size)
            is_first_block = True 
            for i, block_name in enumerate(block_names):
                if (i+1)%FLAGS.block_size!=0 and i!=len(block_names)-1:
                    continue 
                print('HG: i=%d, block_name=%s' %(i, block_name))
                # add l2 losses
                appendix = '_p'+str(kept_percentage)+'_'+str(i)
                collection_name = 'subgraph_losses'+appendix
                # print("HG: collection_name=", collection_name)
                
                outputs = end_points[block_name]
                outputs_pruned = end_points_pruned[block_name]
                l2_loss = add_l2_loss(outputs, outputs_pruned, add_to_collection=True, collection_name=collection_name) 

                # get regularization loss
                if i==len(block_names)-1 and FLAGS.block_size < len(block_names):
                    tmp_block_names = block_names[int(len(block_names)/FLAGS.block_size)*FLAGS.block_size:] 
                    print('HG: last block size:', len(tmp_block_names))
                else:
                    print('HG: this block start from id=', i-FLAGS.block_size+1, ', end before id=', i+1)
                    tmp_block_names = block_names[i-FLAGS.block_size+1:i+1]
                print('HG: this block contains names:', tmp_block_names)

                regularization_losses = get_regularization_losses_with_block_names(net_name_scope_pruned, \
                    tmp_block_names, add_to_collection=True, collection_name=collection_name)
                print_list('regularization_losses', regularization_losses)

                # total loss and its summary
                total_loss = tf.add_n(tf.get_collection(collection_name), name='total_loss')
                for l in tf.get_collection(collection_name)+[total_loss]:
                    tf.summary.scalar(l.op.name+ appendix+'/summary', l)
                total_losses.append(total_loss)

                #############################
                # Add train operation       #
                #############################
                variables_to_train = get_trainable_variables_with_block_names(net_name_scope_pruned, tmp_block_names)
                print_list("variables_to_train", variables_to_train)

                # add train_op
                if is_first_block and kept_percentage==kept_percentages[0]:
                    global_step_tmp = global_step 
                else: 
                    global_step_tmp = tf.Variable(0, trainable=False, name='global_step'+appendix)
                train_op = add_train_op(optimizer, total_loss, global_step_tmp, var_list=variables_to_train)
                is_first_block = False 

                # Gather update_ops: the updates for the batch_norm variables created by network_fn_pruned.
                update_ops = get_update_ops_with_block_names(net_name_scope_pruned, tmp_block_names)
                print_list("update_ops", update_ops)

                update_ops.append(train_op)
                update_op = tf.group(*update_ops)
                with tf.control_dependencies([update_op]):
                    train_tensor = tf.identity(total_loss, name='train_op'+appendix)
                    train_tensors.append(train_tensor)

        # add summary op
        summary_op = tf.summary.merge_all()
        # return 

    
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
            
            if (not FLAGS.continue_training) or (not tf.train.latest_checkpoint(train_dir)):
                ###########################################
                # Restore original model variable values. #
                ###########################################
                variables_to_restore = get_model_variables_within_scopes([net_name_scope_checkpoint+'/'])
                print_list("restore model variables for original", variables_to_restore)
                load_checkpoint(sess, FLAGS.checkpoint_path, var_list=variables_to_restore)

                #################################################
                # Init  pruned networks  with  well-trained model #
                #################################################

                for i in range(len(pruned_net_name_scopes)):
                    net_name_scope_pruned = pruned_net_name_scopes[i]
                    print('net_name_scope_pruned=', net_name_scope_pruned)

                    ## init pruned variables .
                    kept_percentage = kept_percentages[i]
                    prune_info = prune_infos[i]
                    variables_init_value = get_pruned_kernel_matrix(sess, prune_info, net_name_scope_checkpoint)
                    reinit_scopes = [re.sub(net_name_scope_checkpoint, net_name_scope_pruned, name) for name in variables_init_value.keys()]
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
                                          v for v in get_model_variables_within_scopes([net_name_scope_pruned+'/'])
                                          if v not in variables_to_reinit}
                    print_list("restore model variables for "+net_name_scope_pruned, variables_to_restore.values())
                    load_checkpoint(sess, FLAGS.checkpoint_path, var_list=variables_to_restore)
            else:
                # restore all variables from checkpoint
                variables_to_restore = get_global_variables_within_scopes()
                load_checkpoint(sess, train_dir, var_list=variables_to_restore)

            #################################################
            # init unitialized global variable. #
            #################################################
            # uninitialized_variables =[x.decode('utf-8') for x in sess.run(tf.report_uninitialized_variables())]
            # print_list('uninitialized variables', uninitialized_variables)
            # variables_to_init = [v for v in tf.global_variables() if v.name.split(':')[0] in set(uninitialized_variables)]
            # #get_global_variables_within_scopes(uninitialized_variables)
            # print_list("variables_to_init", variables_to_init)
            # sess.run( tf.variables_initializer(variables_to_init) )
            variables_to_init = get_global_variables_within_scopes(sess.run( tf.report_uninitialized_variables() ))
            print_list("init unitialized variables", variables_to_init)
            sess.run( tf.variables_initializer(variables_to_init) )
            
            init_global_step_value = sess.run(global_step)
            print('HG: initial global step: ', init_global_step_value)
            if init_global_step_value >= FLAGS.max_number_of_steps:
                print('Exit: init_global_step_value (%d) >= FLAGS.max_number_of_steps (%d)' \
                    %(init_global_step_value, FLAGS.max_number_of_steps))
                return 

            ###########################
            # Record CPU usage  #
            ###########################
            mpstat_output_filename = os.path.join(train_dir, "cpu-usage.log")
            os.system("mpstat -P ALL 1 > " + mpstat_output_filename + " 2>&1 &")

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
                        results = sess.run([summary_op]+train_tensors)
                        train_summary, loss_values= results[0], results[1:]
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
                    results = sess.run([summary_op]+ total_losses) #loss_value = sess.run(total_loss)
                    train_summary, loss_values = results[0], results[1:]
                    train_writer.add_summary(train_summary, i)
                    format_str = ('%s: step %d, loss = %s')
                    print(format_str % (datetime.now(), i, str(loss_values)))
                    info= format_str % (datetime.now(), i, str(loss_values))
                    write_detailed_info(info)

                # record the evaluation accuracy
                is_last_step = (i==FLAGS.max_number_of_steps)
                if i%FLAGS.evaluate_every_n_steps==0 or is_last_step:

                    # test accuracy; each kept_percentage corresponds to a pruned network, and thus an accuracy. 
                    test_accuracies = []
                    for p in range(len(kept_percentages)):
                        kept_percentage = kept_percentages[p]
                        appendix = '_p'+str(kept_percentage)
                        correct_prediction = correct_predictions[p]
                        # run_meta = (i==FLAGS.evaluate_every_n_steps)&&(p==0)
                        test_accuracy, run_metadata = evaluate_accuracy(sess, coord, test_dataset.num_samples,
                                      test_images, test_labels, test_images, test_labels, 
                                      correct_prediction, FLAGS.test_batch_size, run_meta=False)
                        summary = tf.Summary()
                        summary.value.add(tag='accuracy'+appendix, simple_value=test_accuracy)
                        train_writer.add_summary(summary, i)
                        test_accuracies.append((kept_percentage, test_accuracy))
                    # if run_meta: 
                        # eval_writer.add_run_metadata(run_metadata, 'step%d-eval' % i)
                    acc_str ='['+', '.join(['(%s, %.6f)' %(str(kp), acc) for kp, acc in test_accuracies])+']'
                    info = ('%s: step %d, test_accuracy = %s') % (datetime.now(), i,  str(acc_str))
                    print(info)
                    if i==0 or is_last_step:
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

            train_speed = train_time /train_only_cnt
            train_time = (FLAGS.max_number_of_steps)*train_speed 
            info = "HG: training speed(sec/batch): %.6f\n" %(train_speed)
            info += "HG: training time(min): %.1f, total time(min): %.1f \n" %( train_time/60.0,  total_time/60.0)
            print(info)
            log_info+=info
            write_log_info(log_info)
            write_detailed_info(info)
if __name__ == '__main__':
    tf.app.run()

