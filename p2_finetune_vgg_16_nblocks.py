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



# Modified by Hui Guan,
# phase 2: compose local trained module to make an entire network and fine tune globally. 


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from mpi4py import MPI
import tensorflow as tf
from tensorflow.python.client import timeline

from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
import numpy as np
import re, time,  math , os 
from datetime import datetime
from pprint import pprint
import itertools, random , sys

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

# tf.app.flags.DEFINE_string(
#     'net_name_scope', 'resnet_v1_50',
#     'The name scope of previous trained network in the current graph.')

tf.app.flags.DEFINE_string(
    'net_name_scope_checkpoint', 'vgg_16',
    'The name scope for the saved previous trained network')

tf.app.flags.DEFINE_string(
    'net_name_scope_pruned', 'vgg_16_pruned',
    'The name scope of pruned network in the current graph.')

tf.app.flags.DEFINE_string(
    'kept_percentages', '0.5',
    'The numbers of filters to keep')

tf.app.flags.DEFINE_integer(
    'start_config_id', 0,
    'The start index of configurations to evaluate in the main function')

# tf.app.flags.DEFINE_integer(
#     'num_configurations', 100,
#     'The Number of configurations to evaluate. used when configuration type is "special" or "rank"')

tf.app.flags.DEFINE_integer(
    'total_num_configs', 500,
    'The total number of configurations to evaluate, used to load configurations pickle')

# tf.app.flags.DEFINE_integer(
#     'configuration_index', 0,
#     'The index of configurations to evaluate in the main function')

tf.app.flags.DEFINE_string(
    'config_type', 'special',
    'The way to generate some configurations to evaluate. One of "special", "sample", "rank"')

# tf.app.flags.DEFINE_integer(
#     'local_train_steps', 1000,
#     'The max number of steps used in local train')
tf.app.flags.DEFINE_integer(
    'test_batch_size', 32, 'The number of samples in each batch for test dataset.')

tf.app.flags.DEFINE_string(
    'train_dataset_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'test_dataset_name', 'val', 'The name of the train/test split.')

tf.app.flags.DEFINE_boolean(
    'continue_training', False,
    'if continue training is true, then do not clean the train directory.')

tf.app.flags.DEFINE_integer(
    'max_to_keep', 1, 'The number of models to keep.')


tf.app.flags.DEFINE_integer(
    'block_size', 2,
    'The number of convolutional layers inside a block. The mininum value is 2. ')

tf.app.flags.DEFINE_boolean(
    'last_conv_pruned', False,
    'if true, the last convolutional layer in a block is pruned.')

FLAGS = tf.app.flags.FLAGS


def main(_):
    tic = time.time() 
    tf.logging.set_verbosity(tf.logging.INFO)
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    # initialize constants
    net_name_scope_pruned = FLAGS.net_name_scope_pruned
    net_name_scope_checkpoint = FLAGS.net_name_scope_checkpoint
    kp_options = sorted([float(x) for x in FLAGS.kept_percentages.split(',')])
    num_options = len(kp_options)
    pruned_layer_names = valid_layer_names[:-1]
    num_units = len(pruned_layer_names)
    print_list('kp_options', kp_options)
    print('HG: num_options=%d, num_units=%d' %(num_options, num_units))
    print('HG: total number of configurations=%d' %(num_options**num_units))

    # find the  configurations to evaluate 
    if FLAGS.config_type =='sample':
        configs = get_sampled_configurations(num_units, num_options, FLAGS.total_num_configs)
    elif FLAGS.config_type == 'special':
        configs = get_special_configurations(num_units, num_options)
    num_configs = len(configs)
    print('HG: config_type=', FLAGS.config_type, ', num_configs=', num_configs)

    #Getting MPI rank integer
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank() # use rank as offset
    rank = 0 
    config_id = FLAGS.start_config_id + rank 
    print('HG: start_config_index=%d, rank=%d,  config_index=%d' %(FLAGS.start_config_id, rank,  config_id))
    if config_id >= num_configs:
        print("ERROR: config_id(%d) >= num_configs(%d)" %(config_id, num_configs))
        return
    
    # get the specific configuration 
    config = configs[config_id]
    config = [kp_options[i] for i in config]
    if not FLAGS.last_conv_pruned:
        # if the last conv in a block is not pruned. reset the config
        for i in xrange(len(config)):
            if (i+1)%FLAGS.block_size==0:
                config[i] = 1.0
    print('HG: selected config=', config)
    # return 
    
    # prepare for training with the specific config 
    prune_info={}
    for i in xrange(len(config)):
        layer_name = pruned_layer_names[i]
        prune_info[layer_name]={'kp': config[i]}

    
    # prepare file system 
    if FLAGS.last_conv_pruned:
        foldername = 'last_conv_pruned'
    else:
        foldername = 'last_conv_unpruned'
    results_dir = os.path.join(FLAGS.train_dir, foldername, 'id'+str(config_id)) 
    train_dir = os.path.join(results_dir, 'train')


    if not (FLAGS.continue_training and tf.train.latest_checkpoint(train_dir)):
        prepare_file_system(train_dir)

    def write_detailed_info(info):
        with open(os.path.join(train_dir, 'train_details.txt'), 'a') as f:
            f.write(info+'\n') 

    info = 'train_dir:'+train_dir+'\n'
    info += 'kp_options:'+str(kp_options)+'\n'
    info += 'configuration: '+ str(config)+'\n'
    # print(info)
    write_detailed_info(info)

    with tf.Graph().as_default():
   
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
        network_fn_pruned = nets_factory.get_network_fn_pruned(
            FLAGS.model_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            weight_decay=FLAGS.weight_decay)

        ####################
        # Define the model #
        ####################
        
        logits_train,_ = network_fn_pruned(images, 
                                           prune_info = prune_info, 
                                            is_training=True, 
                                            is_local_train=False, 
                                            reuse_variables=False,
                                            scope = net_name_scope_pruned)

        logits_eval, _ = network_fn_pruned(test_images, 
                                           prune_info = prune_info, 
                                           is_training=False, 
                                           is_local_train=False, 
                                           reuse_variables=True,
                                           scope = net_name_scope_pruned)
        cross_entropy = add_cross_entropy(logits_train, labels)
        correct_prediction = add_correct_prediction(logits_eval, test_labels)

        #############################
        # Specify the loss functions #
        #############################
        collection_name = 'subgraph_losses'
        tf.add_to_collection(collection_name, cross_entropy)
        # get regularization loss
        regularization_losses = get_regularization_losses_within_scopes()
        print_list('regularization_losses', regularization_losses)
        # total loss and its summary
        total_loss = tf.add_n(tf.get_collection(collection_name), name='total_loss')
        for l in tf.get_collection(collection_name)+[total_loss]:
            tf.summary.scalar(l.op.name+'/summary', l)


        #########################################
        # Configure the optimization procedure. #
        #########################################
        with tf.device(deploy_config.variables_device()):
            global_step = tf.Variable(0, trainable=False, name='global_step')
        with tf.device(deploy_config.optimizer_device()):
            learning_rate = configure_learning_rate(dataset.num_samples, global_step, FLAGS)
            optimizer = configure_optimizer(learning_rate, FLAGS)
            tf.summary.scalar('learning_rate', learning_rate)

        #############################
        # Add train operation       #
        #############################
        variables_to_train = get_trainable_variables_within_scopes()
        train_op = add_train_op(optimizer, total_loss, global_step, var_list=variables_to_train)
        print_list("variables_to_train", variables_to_train)

        # Gather update_ops: the updates for the batch_norm variables created by network_fn_pruned.
        update_ops = get_update_ops_within_scopes()
        print_list("update_ops", update_ops)


        update_ops.append(train_op)
        update_op = tf.group(*update_ops)
        with tf.control_dependencies([update_op]):
            train_tensor = tf.identity(total_loss, name='train_op')

        # add summary op
        summary_op = tf.summary.merge_all()


        print("HG: trainable_variables=", len(tf.trainable_variables()))
        print("HG: model_variables=", len(tf.model_variables()))
        print("HG: global_variables=", len(tf.global_variables()))
        # print_list('model_variables but not trainable variables', list(set(tf.model_variables()).difference(tf.trainable_variables())))
        # print_list('global_variables but not model variables', list(set(tf.global_variables()).difference(tf.model_variables())))
        layer_names_dict = {} 
        for layer_name in pruned_layer_names:
            if layer_name in prune_info:
                kp = prune_info[layer_name]
                if kp not in layer_names_dict:
                    layer_names_dict[kp] = [] 
                layer_names_dict[kp].append(layer_name)
        print('HG: layer_names_dict')
        pprint(layer_names_dict)



        sess_config = tf.ConfigProto(intra_op_parallelism_threads=16,
                                        inter_op_parallelism_threads=16)
        with tf.Session(config=sess_config) as sess:
            ###########################
            # prepare for filewritter #
            ###########################
            train_writer = tf.summary.FileWriter(train_dir, sess.graph)

            # if restart the training or there is no checkpoint in the train_dir 
            if FLAGS.continue_training and tf.train.latest_checkpoint(train_dir):
                ###########################################
                ## Restore all variables from checkpoint ##
                ###########################################
                variables_to_restore = get_global_variables_within_scopes()
                load_checkpoint(sess, train_dir, var_list = variables_to_restore)

            else:
                #################################################
                # Restore  pruned model variable values. #
                #################################################

                num_blocks = int(len(valid_layer_names)/FLAGS.block_size)
                block_size = FLAGS.block_size 
                print('HG: block size:', FLAGS.block_size)
                for block_id in xrange(num_blocks):
                    print('HG: number of blocks:', num_blocks, ', block_id:', block_id)
                    start_layer_id = FLAGS.block_size*block_id
                    end_layer_id = start_layer_id+block_size
                    block_layer_names = valid_layer_names[start_layer_id:end_layer_id]
                    if block_id ==0:
                        block_config = config[start_layer_id: end_layer_id]
                    elif block_id == num_blocks-1:
                        block_config = config[start_layer_id-1: end_layer_id-1]
                    else:
                        block_config = config[start_layer_id-1: end_layer_id]
                    print('HG: block_layer_names:', block_layer_names)
                    print('HG: block confiugrations:', block_config)

                    block_config_str = '_'.join(map(str, block_config))
                    checkpoint_path = os.path.join(FLAGS.checkpoint_path, \
                        'm'+str(FLAGS.block_size)+'_b'+str(block_id)+'_'+block_config_str, \
                        'train')

                    train_scopes = [net_name_scope_pruned+'/'+item for item in block_layer_names]
                    variables_to_train = get_model_variables_within_scopes(train_scopes)
                    print_list("restore pruned model variables", variables_to_train)
                    load_checkpoint(sess, checkpoint_path, var_list=variables_to_train)
                    all_variables_to_train.extend(variables_to_train)

                #################################################
                # Restore  orignal  model variable values. #
                #################################################
                variables_to_restore = {re.sub(net_name_scope_pruned, net_name_scope_checkpoint, v.op.name): 
                                      v for v in get_model_variables_within_scopes()
                                      if v not in set(all_variables_to_train)}
                print_list("restore original model variables", variables_to_restore.values())
                load_checkpoint(sess, checkpoint_path, var_list=variables_to_restore)


            #################################################
            # init unitialized global variable. #
            #################################################
            variables_to_init = get_global_variables_within_scopes(sess.run( tf.report_uninitialized_variables() ))
            print_list("init unitialized variables", variables_to_init)
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
            mpstat_output_filename = os.path.join(train_dir, "cpu-usage.log")
            os.system("mpstat -P ALL 1 > " + mpstat_output_filename + " 2>&1 &")

            ###########################
            # Kicks off the training. #
            ###########################
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)
            print('HG: # of threads=', len(threads))


            duration = 0 
            duration_cnt = 0 
            train_time = 0 
            train_only_cnt = 0 

            print("start to train at:", datetime.now())
            for i in range(init_global_step_value, FLAGS.max_number_of_steps+1):
                #train_step = i+FLAGS.local_train_steps
                train_step = i 
                # run optional meta data, or summary, while run train tensor
                if i > init_global_step_value: 
                #if i < FLAGS.max_number_of_steps:
                    
                    # run metadata and train 
                    if i % FLAGS.runmeta_every_n_steps == FLAGS.runmeta_every_n_steps-1:
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()

                        loss_value = sess.run(train_tensor,
                                              options = run_options,
                                              run_metadata=run_metadata)
                        train_writer.add_run_metadata(run_metadata, 'step%d-train' % i)

                        # Create the Timeline object, and write it to a json file
                        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                        chrome_trace = fetched_timeline.generate_chrome_trace_format()
                        with open(os.path.join(train_dir, 'timeline_'+str(i)+'.json'), 'w') as f:
                            f.write(chrome_trace)

                    # record summary and train 
                    elif i % FLAGS.summary_every_n_steps==0:
                        train_summary, loss_value = sess.run([summary_op, train_tensor])
                        train_writer.add_summary(train_summary, train_step)

                    # train only 
                    else:
                        start_time = time.time()
                        loss_value = sess.run(train_tensor)
                        train_only_cnt+=1
                        train_time += time.time() - start_time 
                        duration_cnt +=1 
                        duration += time.time()- start_time 

                    if i%FLAGS.log_every_n_steps==0 and duration_cnt > 0:
                        log_frequency = duration_cnt  
                        examples_per_sec = log_frequency * FLAGS.batch_size / duration
                        sec_per_batch = float(duration /log_frequency)
                        summary = tf.Summary()
                        summary.value.add(tag='examples_per_sec', simple_value=examples_per_sec)
                        summary.value.add(tag='sec_per_batch', simple_value=sec_per_batch)
                        train_writer.add_summary(summary, train_step)
                        format_str = ('%s: step %d, loss = %s (%.1f examples/sec; %.3f sec/batch)')
                        print(format_str % (datetime.now(), i, str(loss_value), examples_per_sec, sec_per_batch))
                        duration = 0
                        duration_cnt = 0  

                        info= format_str % (datetime.now(), i, str(loss_value), examples_per_sec, sec_per_batch)
                        write_detailed_info(info)
                else:
                    # run only total loss when i=0 
                    train_summary, loss_value = sess.run([summary_op, total_loss]) #loss_value = sess.run(total_loss)
                    train_writer.add_summary(train_summary, train_step)
                    format_str = ('%s: step %d, loss = %s')
                    print(format_str % (datetime.now(), i, str(loss_value)))
                    info= format_str % (datetime.now(), i, str(loss_value))
                    write_detailed_info(info)

                # record the evaluation accuracy
                is_last_step = (i==FLAGS.max_number_of_steps)
                if i%FLAGS.evaluate_every_n_steps==0 or is_last_step:

                    test_accuracy, run_metadata = evaluate_accuracy(sess, coord, test_dataset.num_samples,
                                  test_images, test_labels, test_images, test_labels, 
                                  correct_prediction, FLAGS.test_batch_size, run_meta=False)
                    summary = tf.Summary()
                    summary.value.add(tag='accuracy', simple_value=test_accuracy)
                    train_writer.add_summary(summary,train_step)

                    info = ('%s: step %d, test_accuracy = %.6f') % (datetime.now(), train_step,  test_accuracy)
                    print(info)
                    write_detailed_info(info)

                    ###########################
                    # Save model parameters . #
                    ###########################
                    save_path = saver.save(sess, os.path.join(train_dir, 'model.ckpt-'+str(i)))
                    print("HG: Model saved in file: %s" % save_path)

            coord.request_stop()
            coord.join(threads)
            total_time = time.time()-tic 
#            train_time = train_time*(FLAGS.max_number_of_steps - init_global_step_value)/train_only_cnt
#            info = "HG: training time(min): %.1f, total time(min): %.1f \n" %( train_time/60.0,  total_time/60.0)
            train_speed = train_time*1.0/train_only_cnt
            train_time = train_speed*(FLAGS.max_number_of_steps) #- init_global_step_value) #/train_only_cnt
            info = "HG: training speed(sec/batch): %.6f\n" %(train_speed)
            info += "HG: training time(min): %.1f, total time(min): %.1f" %( train_time/60.0,  total_time/60.0)
            print(info)
            write_detailed_info(info)

            # ###########################
            # # Save model parameters . #
            # ###########################
            #saver = tf.train.Saver()
            #save_path = saver.save(sess, os.path.join(train_dir, 'model.ckpt'), i)
            #print("HG: Model saved in file: %s \n" % save_path)

                
if __name__ == '__main__':
    tf.app.run()
