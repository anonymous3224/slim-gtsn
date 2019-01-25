# train_helper.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os 
from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
import numpy as np
import re, time,  math 
from datetime import datetime
from pprint import pprint
import itertools, random , sys

import cPickle as pickle

import tensorflow as tf
slim = tf.contrib.slim

# picke load and save 
def check_directory(path):
  if not os.path.isdir(path):
    os.makedirs(path)

def save_pickle(save_path, save_name, save_object):
  check_directory(save_path)
  filepath = os.path.join(save_path, save_name)
  pickle.dump(save_object, open(filepath,"wb" ))
  print('File saved to:', filepath)

def load_pickle(load_path, load_name=None):
  if load_name:
    filepath =  os.path.join(load_path, load_name)
  else:
    filepath = load_path 
  print('Load pickle file:', filepath)
  return pickle.load( open(filepath, "rb" ))


def configure_learning_rate(num_samples_per_epoch, global_step, FLAGS):
  """Configures the learning rate.

  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """
  print('HG: num_samples_per_epoch=', num_samples_per_epoch, ", batch_size=", FLAGS.batch_size, 
    ", num_epochs_per_decay=", FLAGS.num_epochs_per_decay)
  decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *FLAGS.num_epochs_per_decay)
  if FLAGS.sync_replicas:
    decay_steps /= FLAGS.replicas_to_aggregate

  if FLAGS.learning_rate_decay_type == 'exponential':
    print('HG: exponential learning rate with lr=', FLAGS.learning_rate, ', decay_steps=', decay_steps, 
      'learning_rate_decay_factor=', FLAGS.learning_rate_decay_factor)
    return tf.train.exponential_decay(FLAGS.learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'fixed':
    print('HG: fixed learning rate with lr=', FLAGS.learning_rate)
    return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'polynomial':
    print('HG: polynomial learning rate with lr=', FLAGS.learning_rate, ', decay_steps=', decay_steps, 
      'end_learning_rate=', FLAGS.end_learning_rate)
    return tf.train.polynomial_decay(FLAGS.learning_rate,
                                     global_step,
                                     decay_steps,
                                     FLAGS.end_learning_rate,
                                     power=1.0,
                                     cycle=False,
                                     name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized',
                     FLAGS.learning_rate_decay_type)


def configure_optimizer(learning_rate, FLAGS):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
  """
  print('HG: optimizer_type=', FLAGS.optimizer)
  if FLAGS.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=FLAGS.adadelta_rho,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=FLAGS.ftrl_learning_rate_power,
        initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
        l1_regularization_strength=FLAGS.ftrl_l1,
        l2_regularization_strength=FLAGS.ftrl_l2)
  elif FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        name='Momentum')
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.rmsprop_momentum,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
  return optimizer


#############################
# define graph components  #
#############################

def train_inputs(dataset, deploy_config, FLAGS):
  #####################################
  # Select the preprocessing function #
  #####################################
  preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
  image_preprocessing_fn = preprocessing_factory.get_preprocessing(
      preprocessing_name,
      is_training=True)
  # print('HG: preprocessing_name=', preprocessing_name)
  # print('HG: image_preprocessing_fn=', image_preprocessing_fn)
  
  ##############################################################
  # Create a dataset provider that loads data from the dataset #
  ##############################################################
  # print('HG: inputs_device=', deploy_config.inputs_device())
  # print('HG: FLAGS.num_readers=', FLAGS.num_readers)
  # print('HG: FLAGS.batch_size=', FLAGS.batch_size)
  with tf.device(deploy_config.inputs_device()):
    with tf.name_scope('train_data_provider'):
      provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=FLAGS.num_readers,
        common_queue_capacity=20 * FLAGS.batch_size,
        common_queue_min=10 * FLAGS.batch_size)
      [image, label] = provider.get(['image', 'label'])
      label -= FLAGS.labels_offset
    # print('HG: image, label=', image, label)

    train_image_size = FLAGS.train_image_size #or network_fn.default_image_size
    with tf.name_scope('train_preprocessing'):
      image = image_preprocessing_fn(image, train_image_size, train_image_size)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)
    # labels = slim.one_hot_encoding(
    #     labels, dataset.num_classes - FLAGS.labels_offset)

    batch_queue = slim.prefetch_queue.prefetch_queue(
          [images, labels], capacity=2 * deploy_config.num_clones)
  return batch_queue


def test_inputs(dataset, deploy_config, FLAGS):
  #####################################
  # Select the preprocessing function #
  #####################################
  preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
  image_preprocessing_fn = preprocessing_factory.get_preprocessing(
      preprocessing_name,
      is_training=False)

  ##############################################################
  # Create a dataset provider that loads data from the dataset #
  ##############################################################
  with tf.device(deploy_config.inputs_device()):
    with tf.name_scope('test_data_provider'):
      provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=10 * FLAGS.test_batch_size,
        common_queue_min=2 * FLAGS.test_batch_size)
      [image, label] = provider.get(['image', 'label'])
      label -= FLAGS.labels_offset

    eval_image_size = FLAGS.train_image_size #or network_fn.default_image_size
    with tf.name_scope('test_preprocessing'):
      image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=FLAGS.test_batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.test_batch_size)
  return images, labels 



def add_cross_entropy(logits, labels):
	# Calculate the average cross entropy loss across the batch.
	# WARNING: This op expects unscaled logits, since it performs a softmax on logits internally for efficiency. 
	# Do not call this op with the output of softmax, as it will produce incorrect results.
	labels = tf.cast(labels, tf.int64)
	with tf.name_scope('cross_entropy'):
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
              labels=labels, logits=logits, name='cross_entropy_per_example')
		cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	return cross_entropy_mean


def add_correct_prediction(logits, labels):
	labels = tf.cast(labels, tf.int64)
	with tf.name_scope('correct_prediction'):
		results_tensor = tf.nn.softmax(logits, name='softmax')
		predictions = tf.argmax(results_tensor, 1)
		correct_prediction = tf.equal(predictions, labels)
	return correct_prediction

def add_l2_loss(output, gnd_output, weights=1.0, add_to_collection = True, collection_name='subgraph_losses'):
  # loss function with l2_norm
  # l2_loss = tf.norm(tf.reshape(output - gnd_output, [FLAGS.batch_size, -1]), axis=1,name="l2_loss_per_example")
  # l2_loss_mean = tf.reduce_mean(l2_loss, name='l2_loss')
  # tf.add_to_collection('subgraph_losses', l2_loss_mean)
  # return l2_loss_mean 

  # loss function with l2_loss
  shape = output.get_shape().as_list()
  size = np.prod(shape)
  l2_loss = tf.nn.l2_loss(output - gnd_output, name="raw_activation_loss")
  l2_loss = tf.div(l2_loss*1.0, size*1.0, name="activation_loss") 
  l2_loss = tf.multiply(l2_loss, weights, name="scaled_activation_loss")
  if add_to_collection:  
    tf.add_to_collection(collection_name, l2_loss)
  return l2_loss

def add_train_op(optimizer, total_loss, global_step, var_list):

  grads = optimizer.compute_gradients(total_loss, var_list=var_list)
  train_op = optimizer.apply_gradients(grads, global_step=global_step)

  # add summary for grads, var
  with tf.name_scope('norm'):
    for grad, var in grads:
      if grad is not None:
        #tf.summary.histogram(var.op.name+'/gradients', grad)
        tf.summary.scalar(var.op.name+'/gradients', tf.norm(grad))
        tf.summary.scalar(var.op.name+'/norm', tf.norm(var))
        # tf.summary.scalar(var.op.name+'/mean', tf.reduce_mean(var))
        # tf.summary.histogram(var.op.name+'/gradients', grad)
      else:
        print('trainable variable has no gradient. var=', var)

  return train_op

def evaluate_accuracy(sess, coord, num_samples,
                          test_images, test_labels, 
                          images, labels, 
                          correct_prediction, batch_size,
                          run_meta = False):

  # evaluate the trained model using testing dataset
  num_iter = int(math.ceil(num_samples*1.0/ batch_size))
  true_count = 0 # Counts the number of correct predictions.
  total_sample_count = num_iter * batch_size
  step = 0
  run_metadata = None 
  print('HG: num_iter=%d, num_samples=%d, batch_size=%d, total_sample_count=%d' %(num_iter, num_samples, batch_size, total_sample_count))
  if run_meta:
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
  while step < num_iter and not coord.should_stop():
    test_images_value, test_labels_value = sess.run([test_images, test_labels])
    if step ==int(num_iter/2) and run_meta == True:
      prediction_value = sess.run([correct_prediction],
                                  feed_dict={images: test_images_value,
                                             labels: test_labels_value},
                                  options=run_options,
                                  run_metadata=run_metadata)
    else:
      prediction_value = sess.run([correct_prediction],
                                  feed_dict={images: test_images_value,
                                             labels: test_labels_value})
    true_count += np.sum(prediction_value)
    #print('HG: prediction_value:', prediction_value, 'true_count:', true_count)
    step +=1
  accuracy = true_count*1.0/total_sample_count
  return  accuracy, run_metadata

def variable_summaries(var, name_scope=None):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  if name_scope == None:
    name_scope = var.op.name+'/summary'
  with tf.name_scope(name_scope):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    


#############################
# file IO: load, create, etc #
#############################

def directory_contains_filename_start_with_string(directory, filename):
    files = os.listdir(directory)
    for f in files:
        if filename in f:
            return True
    return False 

def load_checkpoint(sess, checkpoint_path, var_list=None):
  if var_list == None:
    saver = tf.train.Saver()
  elif len(var_list)==0:
    print('HG: no variables need to restore from the checkpoint')
    return 
  else:
    saver = tf.train.Saver(var_list)

  print("HG: try to load checkpoint from path:", checkpoint_path) 

  # if it is a directory, find the exact path 
  if tf.gfile.IsDirectory(checkpoint_path):
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
      checkpoint_path = ckpt.model_checkpoint_path
    else:
      raise IOError('checkpoint file not found')

  # load checkpoint from the exact path
  try:
    saver.restore(sess, checkpoint_path)
  except:
    print('ERROR: unable to restore from checkpoint_path:', checkpoint_path)
    return False

  global_step = checkpoint_path.split('/')[-1].split('-')[-1]
  print('HG: load checkpoint with global_step=',global_step)
  return True


def prepare_file_system(directory):
	# Setup the directory we'll write summaries to for TensorBoard
	if tf.gfile.Exists(directory):
		tf.gfile.DeleteRecursively(directory)
	tf.gfile.MakeDirs(directory)
	return


###################################
# Qeury variables based on scopes #
###################################

def get_regularization_losses_within_scopes(scopes=None, add_to_collection = True, collection_name='subgraph_losses'):
	losses = []
	if scopes is None:
		losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	else:
		for scope in scopes:
			losses.extend(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope))
		losses = list(set(losses))# avoid duplicated items
	if add_to_collection:
		total_regularization_loss = tf.add_n(losses, name='regularization_loss')
		tf.add_to_collection(collection_name, total_regularization_loss)
	return losses

def get_update_ops_within_scopes(scopes=None):
	if scopes is None:
		return tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	update_ops = []
	for scope in scopes:
		update_ops.extend(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
	return list(set(update_ops))

def get_trainable_variables_within_scopes(scopes=None):
	if scopes is None:
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
	variables_to_prune = []
	for scope in scopes:
		variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
		variables_to_prune.extend(variables)
	return list(set(variables_to_prune)) 

def get_model_variables_within_scopes(scopes=None):
	if scopes is None:
		return tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
	variables_to_prune = []
	for scope in scopes:
		variables = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope)
		variables_to_prune.extend(variables)
	return list(set(variables_to_prune))

def get_global_variables_within_scopes(scopes=None):
	if scopes is None:
		return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) 
	variables_to_prune = []
	for scope in scopes:
		variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)
		variables_to_prune.extend(variables)
	return list(set(variables_to_prune))



###################################
# Functions for Pruning variables  #
###################################

def find_weights_index(variables):
  for i, v in enumerate(variables):
    if 'weights' in v.op.name:
      return i
  return 

def find_all_weights_index(variables):
  indexes = [] 
  for i, v in enumerate(variables):
    if 'weights' in v.op.name:
      indexes.append(i)
  return indexes 

def get_pruned_filter_indexes(weights, kept_percentage=0.5):
  # remove the filters with the smallest l1/l2 norm in the weights
  weights_shape = weights.shape
  # print('HG: weights_shape=', weights_shape)

  num_filters = weights_shape[-1]
  num_filters_kept = int(num_filters*kept_percentage)
  num_filters_pruned = num_filters - num_filters_kept
  #print('HG: kept_percentage, num_filters_pruned=', (kept_percentage, num_filters_pruned))

  weights_norm = np.linalg.norm(np.reshape(weights, [-1, weights_shape[-1]]), ord=1, axis=0)
  #print('HG: weights_norm=', weights_norm)
      
  weights_norm_normalized = weights_norm/np.max(weights_norm)
  #print('HG: weights_norm_normalized=', weights_norm_normalized)
  norm_sorted = sorted(zip(range(len(weights_norm)), 
          weights_norm_normalized), key=lambda a: a[-1])
  #print('HG: norm_sorted=',norm_sorted)

  filter_indexes =[item[0] for item in norm_sorted[:num_filters_pruned]]
  # print('HG: filter_indexes=', len(filter_indexes), filter_indexes)
  return filter_indexes


###################################
# Functions for exploring configs #
###################################

def get_kept_percentages_dict_from_path(path):
    '''return a dictionay with key=kept_percentages, value=folders that contains the kept_percentage'''
    folders = os.listdir(path)
    kps = {}
    for folder in folders:
        matches = re.findall(r'kp(\d\.\d+)', folder)
        if len(matches):
            kp = float(matches[0])
            #steps = folder.strip().split('_')[-1]
            if kp not in kps:
                kps[kp]=[]
            kps[kp].append(folder)
    return kps



def get_sampled_configurations(num_units, num_options, num_configs):
  save_path = './configs_enum'
  save_name = 'configs_%d_units_%d_options_%d.p' %(num_configs, num_units, num_options)
  configs = load_pickle(save_path, save_name)
  return configs 



def get_special_configurations(num_units, num_options):
    '''get a set of special configurations. e.g.: (0, 0, 0), (1, 1, 1)'''
    special_configs = [list([x]*num_units) for x in range(num_options)]
    return special_configs




def print_list(name, l, top=5):
  print('HG:', name, ', len=', len(l))
  if top==0 or top>len(l):
    top = len(l)
  cnt = 0
  for item in l:
    print('\t', item)
    cnt +=1
    if cnt >=top:
      break 
  if top < len(l):
    print('\t ...')


if __name__ == '__main__':
	# print(indexed_prune_scopes_to_prune_scopes(['111']))
	# print(indexed_prune_scopes_to_shorten_scopes(['111']))
	# print(sorted(valid_indexed_prune_scopes()))
	# prune_info={1:{1:{'inputs': None, 'prune_layers':{1: 0.5}}}}
	# set_prune_units_inputs(prune_info, 'resnet_v1_50')
	# print(prune_info) 
	# print(get_prune_units_outputs_scope('111'))
  
  num_units = 33
  num_options = 3
  n = 500
  configs = get_sampled_configurations(num_units, num_options, n)
  configs2 = get_sampled_configurations(num_units, num_options, n) 
  print_list('configs', configs)
  print_list('configs2', configs2)
  # print(_get_num_samples_per_sum_upper_bounds(num_units, num_options, n))






