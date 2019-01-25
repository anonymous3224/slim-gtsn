# train filter for resnet_v1_50 
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

from train_helper import *
import tensorflow as tf
slim = tf.contrib.slim



###################################
# Functions for Pruning variables  #
###################################


def get_pruned_kernel_matrix(sess, prune_scopes, shorten_scopes, kept_percentages=0.5):
	''' get the init values for pruned kernels in a greedy way'''
	init_values = {}
	pruned_filter_indexes = {} 
	if not isinstance(prune_scopes, list):
		prune_scopes = [prune_scopes]
		shorten_scopes = [shorten_scopes]

	if not isinstance(kept_percentages, list):
		kept_percentages = [kept_percentages]*len(prune_scopes)

	for si in range(len(prune_scopes)):
		prune_scope = prune_scopes[si]
		shorten_scope = shorten_scopes[si]
		kept_percentage = kept_percentages[si]

		variables_to_prune = get_model_variables_within_scopes([prune_scope])
		variables_to_shorten = get_model_variables_within_scopes([shorten_scope])

		# find the weights kernel from variables to prune
		index = find_weights_index(variables_to_prune)
		if variables_to_prune[index].op.name in init_values:
			variables_value = [init_values[variable_to_prune.op.name] for variable_to_prune in variables_to_prune]
		else:
			variables_value = sess.run(variables_to_prune) 
		weights = variables_value[index]

		# calculate the initial values for need-to-change variables
		filter_indexes = get_pruned_filter_indexes(weights, kept_percentage)
		new_weights = np.delete(weights, filter_indexes, axis=3)
		pruned_filter_indexes[prune_scope] = filter_indexes 
		# print('HG: new_weights.shape=', new_weights.shape)
	 
		# pruned layer
		for i in range(len(variables_to_prune)):
			if i == index:
				new_value = new_weights
				print('HG: pruned layers:', variables_to_prune[i].op.name, variables_value[i].shape, '-->' ,new_value.shape)
			else:
				new_value = np.delete(variables_value[i], filter_indexes, axis=0)
			init_values[variables_to_prune[i].op.name] = new_value

		# the layer followed by the pruned layer
		index = find_weights_index(variables_to_shorten)
		variables_value = sess.run(variables_to_shorten)
		for i in range(len(variables_to_shorten)):
			if i == index:
				new_value = np.delete(variables_value[i], filter_indexes, axis=2)
				#print('HG: shorten layers:', variables_to_shorten[i].op.name, variables_value[i].shape, '-->' ,new_value.shape)
			else:
				new_value = variables_value[i]
			
			init_values[variables_to_shorten[i].op.name] = new_value
	return init_values, pruned_filter_indexes



# name of the layers that are valid for pruning. 
valid_layer_names=[ \
	'conv1/conv1_1', 'conv1/conv1_2', \
	'conv2/conv2_1', 'conv2/conv2_2', \
	'conv3/conv3_1', 'conv3/conv3_2', 'conv3/conv3_3', \
	'conv4/conv4_1', 'conv4/conv4_2', 'conv4/conv4_3', \
	'conv5/conv5_1', 'conv5/conv5_2', 'conv5/conv5_3', \
	'fc6', 'fc7', 'fc8']


all_layer_names = [\
	'conv1/conv1_1', 'conv1/conv1_2', 'pool1', \
	'conv2/conv2_1', 'conv2/conv2_2', 'pool2', \
	'conv3/conv3_1', 'conv3/conv3_2', 'conv3/conv3_3', 'pool3', \
	'conv4/conv4_1', 'conv4/conv4_2', 'conv4/conv4_3', 'pool4',\
	'conv5/conv5_1', 'conv5/conv5_2', 'conv5/conv5_3', 'pool5',\
	'fc6', 'dropout6','fc7','dropout7', 'fc8'] 

def layer_name_to_prune_scope(layer_name, net_name_scope):
	return net_name_scope+'/'+layer_name 

def layer_name_to_shorten_scope(layer_name, net_name_scope):
	shorten_layer_index = valid_layer_names.index(layer_name)+1
	if shorten_layer_index >= len(valid_layer_names):
		return None #net_name_scope+'/'+'fc8'
	else:
		return net_name_scope+'/'+valid_layer_names[shorten_layer_index]


###################################
# Functions for Pruning Scopes  #
###################################

def add_activation_maps_loss(output, output_pruned, add_to_collection = True, collection_name='subgraph_losses'):
  # loss function with l2_norm
  # l2_loss = tf.norm(tf.reshape(output - gnd_output, [FLAGS.batch_size, -1]), axis=1,name="l2_loss_per_example")
  # l2_loss_mean = tf.reduce_mean(l2_loss, name='l2_loss')
  # tf.add_to_collection('subgraph_losses', l2_loss_mean)
  # return l2_loss_mean 

  # loss function with l2_loss
  shape = output.get_shape().as_list()
  num_maps = shape[-1]
  print('HG: output shape:', shape)#[32, 224, 224, 64]

  shape_pruned = output_pruned.get_shape().as_list()
  num_maps_pruned = shape_pruned[-1]
  print('HG: output_pruned shape:', shape_pruned) # [32, 224, 224, 19]

  
  output_3 = tf.reshape(output, [shape[0], -1, shape[-1]]) # [32, 224*224, 64]
  output_pruned_3 = tf.reshape(output_pruned, [shape_pruned[0], -1, shape_pruned[-1]]) # [32, 224*224, 16]
  print('HG: reshape output shape:', output_3.get_shape().as_list())
  print('HG: reshape output_pruned shape:', output_pruned_3.get_shape().as_list())
  
  # get the top num_activation_pruned out of num_activations based on l1/norm or l2_norm
  output_3_norm = tf.norm(output_3, ord=1, axis=1) #[32, 64]
  print('HG: output_3_norm shape:', output_3_norm.get_shape().as_list())
  _, indexes = tf.nn.top_k(output_3_norm, k=num_maps_pruned, sorted=False)
  output_3_gnd = tf.stack([tf.stack([output_3[i,:, indexes[i,j]] for j in range(num_maps_pruned)], axis=-1) for i in range(shape[0])], axis=0)
  print('HG: output_3_gnd shape:',  output_3_gnd.get_shape().as_list())

  with tf.name_scope('activation_maps_summaries'):
    tf.summary.scalar(output.op.name+'/mean', tf.reduce_mean(output_3_gnd))
    tf.summary.scalar(output.op.name+'/max', tf.reduce_max(output_3_gnd))
    tf.summary.scalar(output_pruned.op.name+'/mean', tf.reduce_mean(output_pruned))
    tf.summary.scalar(output_pruned.op.name+'/max', tf.reduce_max(output_pruned))
  

  size = np.prod(shape_pruned)
  l2_loss = tf.nn.l2_loss(output_pruned_3 - output_3_gnd, name="raw_activation_loss")
  l2_loss = tf.div(l2_loss*1.0, size*1.0, name="activation_loss") 
  if add_to_collection:  
    tf.add_to_collection(collection_name, l2_loss)
  return l2_loss



# endpoints:
# ('vgg_16/conv1/conv1_1', <tf.Tensor 'vgg_16/conv1/conv1_1/Relu:0' shape=(32, 224, 224, 64) dtype=float32>)
# ('vgg_16/conv1/conv1_2', <tf.Tensor 'vgg_16/conv1/conv1_2/Relu:0' shape=(32, 224, 224, 64) dtype=float32>)
# ('vgg_16/pool1', <tf.Tensor 'vgg_16/pool1/MaxPool:0' shape=(32, 112, 112, 64) dtype=float32>)
# ('vgg_16/conv2/conv2_1', <tf.Tensor 'vgg_16/conv2/conv2_1/Relu:0' shape=(32, 112, 112, 128) dtype=float32>)
# ('vgg_16/conv2/conv2_2', <tf.Tensor 'vgg_16/conv2/conv2_2/Relu:0' shape=(32, 112, 112, 128) dtype=float32>)
# ('vgg_16/pool2', <tf.Tensor 'vgg_16/pool2/MaxPool:0' shape=(32, 56, 56, 128) dtype=float32>)
# ('vgg_16/conv3/conv3_1', <tf.Tensor 'vgg_16/conv3/conv3_1/Relu:0' shape=(32, 56, 56, 256) dtype=float32>)
# ('vgg_16/conv3/conv3_2', <tf.Tensor 'vgg_16/conv3/conv3_2/Relu:0' shape=(32, 56, 56, 256) dtype=float32>)
# ('vgg_16/conv3/conv3_3', <tf.Tensor 'vgg_16/conv3/conv3_3/Relu:0' shape=(32, 56, 56, 256) dtype=float32>)
# ('vgg_16/pool3', <tf.Tensor 'vgg_16/pool3/MaxPool:0' shape=(32, 28, 28, 256) dtype=float32>)
# ('vgg_16/conv4/conv4_1', <tf.Tensor 'vgg_16/conv4/conv4_1/Relu:0' shape=(32, 28, 28, 512) dtype=float32>)
# ('vgg_16/conv4/conv4_2', <tf.Tensor 'vgg_16/conv4/conv4_2/Relu:0' shape=(32, 28, 28, 512) dtype=float32>)
# ('vgg_16/conv4/conv4_3', <tf.Tensor 'vgg_16/conv4/conv4_3/Relu:0' shape=(32, 28, 28, 512) dtype=float32>)
# ('vgg_16/pool4', <tf.Tensor 'vgg_16/pool4/MaxPool:0' shape=(32, 14, 14, 512) dtype=float32>)
# ('vgg_16/conv5/conv5_1', <tf.Tensor 'vgg_16/conv5/conv5_1/Relu:0' shape=(32, 14, 14, 512) dtype=float32>)
# ('vgg_16/conv5/conv5_2', <tf.Tensor 'vgg_16/conv5/conv5_2/Relu:0' shape=(32, 14, 14, 512) dtype=float32>)
# ('vgg_16/conv5/conv5_3', <tf.Tensor 'vgg_16/conv5/conv5_3/Relu:0' shape=(32, 14, 14, 512) dtype=float32>)
# ('vgg_16/pool5', <tf.Tensor 'vgg_16/pool5/MaxPool:0' shape=(32, 7, 7, 512) dtype=float32>)
# ('vgg_16/fc6', <tf.Tensor 'vgg_16/fc6/Relu:0' shape=(32, 1, 1, 4096) dtype=float32>)
# ('vgg_16/dropout6', <tf.Tensor 'vgg_16/dropout6/Identity:0' shape=(32, 1, 1, 4096) dtype=float32>)
# ('vgg_16/fc7', <tf.Tensor 'vgg_16/fc7/Relu:0' shape=(32, 1, 1, 4096) dtype=float32>)
# ('vgg_16/dropout7', <tf.Tensor 'vgg_16/dropout7/Identity:0' shape=(32, 1, 1, 4096) dtype=float32>)
# ('vgg_16/fc8', <tf.Tensor 'vgg_16/fc8/squeezed:0' shape=(32, 200) dtype=float32>)
