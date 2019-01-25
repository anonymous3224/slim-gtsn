# train helper for inception_v3 

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
# Functions for setting network  #
###################################

valid_block_names = ['Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c',
      'Mixed_6d', 'Mixed_6e', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c']

num_units = 11

def config_to_kept_percentage_sequence(config, block_names, kept_percentages):
	if len(config) != len(block_names):
		print('len(config)', len(config))
		print('len(block_names)', len(block_names))
		raise ValueError(' len(config)!= len(block_names)')
	return [kept_percentages[option] for option in config] 


def kept_percentage_sequence_to_prune_info(kept_percentage, block_names):
	# each block name is a key, the value is a dictionary contains two keys: inputs and kp 
	if not isinstance(kept_percentage, list):
		kept_percentage = [kept_percentage]*len(block_names)
	prune_info = {}
	for kp, block_name in zip(kept_percentage, block_names):
		prune_info[block_name]={}
		prune_info[block_name]['kp'] = kp 
	return prune_info 

## all the endpoints for inception_v3 

# {'AuxLogits': <tf.Tensor 'InceptionV3_pruned/AuxLogits/SpatialSqueeze:0' shape=(32, 200) dtype=float32>,
#  'Conv2d_1a_3x3': <tf.Tensor 'InceptionV3_pruned/InceptionV3_pruned/Conv2d_1a_3x3/Relu:0' shape=(32, 111, 111, 32) dtype=float32>,
#  'Conv2d_2a_3x3': <tf.Tensor 'InceptionV3_pruned/InceptionV3_pruned/Conv2d_2a_3x3/Relu:0' shape=(32, 109, 109, 32) dtype=float32>,
#  'Conv2d_2b_3x3': <tf.Tensor 'InceptionV3_pruned/InceptionV3_pruned/Conv2d_2b_3x3/Relu:0' shape=(32, 109, 109, 64) dtype=float32>,
#  'Conv2d_3b_1x1': <tf.Tensor 'InceptionV3_pruned/InceptionV3_pruned/Conv2d_3b_1x1/Relu:0' shape=(32, 54, 54, 80) dtype=float32>,
#  'Conv2d_4a_3x3': <tf.Tensor 'InceptionV3_pruned/InceptionV3_pruned/Conv2d_4a_3x3/Relu:0' shape=(32, 52, 52, 192) dtype=float32>,
#  'Logits': <tf.Tensor 'InceptionV3_pruned/Logits/SpatialSqueeze:0' shape=(32, 200) dtype=float32>,
#  'MaxPool_3a_3x3': <tf.Tensor 'InceptionV3_pruned/InceptionV3_pruned/MaxPool_3a_3x3/MaxPool:0' shape=(32, 54, 54, 64) dtype=float32>,
#  'MaxPool_5a_3x3': <tf.Tensor 'InceptionV3_pruned/InceptionV3_pruned/MaxPool_5a_3x3/MaxPool:0' shape=(32, 25, 25, 192) dtype=float32>,
#  'Mixed_5b': <tf.Tensor 'InceptionV3_pruned/InceptionV3_pruned/Mixed_5b/concat:0' shape=(32, 25, 25, 256) dtype=float32>,
#  'Mixed_5c': <tf.Tensor 'InceptionV3_pruned/InceptionV3_pruned/Mixed_5c/concat:0' shape=(32, 25, 25, 288) dtype=float32>,
#  'Mixed_5d': <tf.Tensor 'InceptionV3_pruned/InceptionV3_pruned/Mixed_5d/concat:0' shape=(32, 25, 25, 288) dtype=float32>,
#  'Mixed_6a': <tf.Tensor 'InceptionV3_pruned/InceptionV3_pruned/Mixed_6a/concat:0' shape=(32, 12, 12, 768) dtype=float32>,
#  'Mixed_6b': <tf.Tensor 'InceptionV3_pruned/InceptionV3_pruned/Mixed_6b/concat:0' shape=(32, 12, 12, 768) dtype=float32>,
#  'Mixed_6c': <tf.Tensor 'InceptionV3_pruned/InceptionV3_pruned/Mixed_6c/concat:0' shape=(32, 12, 12, 768) dtype=float32>,
#  'Mixed_6d': <tf.Tensor 'InceptionV3_pruned/InceptionV3_pruned/Mixed_6d/concat:0' shape=(32, 12, 12, 768) dtype=float32>,
#  'Mixed_6e': <tf.Tensor 'InceptionV3_pruned/InceptionV3_pruned/Mixed_6e/concat:0' shape=(32, 12, 12, 768) dtype=float32>,
#  'Mixed_7a': <tf.Tensor 'InceptionV3_pruned/InceptionV3_pruned/Mixed_7a/concat:0' shape=(32, 5, 5, 1280) dtype=float32>,
#  'Mixed_7b': <tf.Tensor 'InceptionV3_pruned/InceptionV3_pruned/Mixed_7b/concat:0' shape=(32, 5, 5, 2048) dtype=float32>,
#  'Mixed_7c': <tf.Tensor 'InceptionV3_pruned/InceptionV3_pruned/Mixed_7c/concat:0' shape=(32, 5, 5, 2048) dtype=float32>,
#  'PreLogits': <tf.Tensor 'InceptionV3_pruned/Logits/Dropout_1b/dropout/mul:0' shape=(32, 1, 1, 2048) dtype=float32>,
#  'Predictions': <tf.Tensor 'InceptionV3_pruned/Predictions/Reshape_1:0' shape=(32, 200) dtype=float32>}


def set_prune_info_inputs(prune_info, end_points=None, block_size=1):
	for block_name, block_info in prune_info.items():
		if block_name == 'Mixed_5b':
			inputs_key = 'MaxPool_5a_3x3'
		else:
			building_block_id = valid_block_names.index(block_name)
			if building_block_id%block_size!=0:
				continue 
			inputs_key = valid_block_names[valid_block_names.index(block_name)-1]
		block_info['inputs'] = inputs_key if end_points is None else end_points[inputs_key]

###################################
# Qeury variables based on scopes #
###################################
# since variables names starts with the string InceptionV3, but the regularization loss 
# and update op starts with two string InceptionV3. The get variables or operations function
# are rewritten for InceptionV3 using keyword matching instead of scope search. 

def get_regularization_losses_with_block_names(net_name_scope='InceptionV3_pruned', 
											   block_names=None, 
											   add_to_collection = True, 
											   collection_name='subgraph_losses'):
	
	all_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	losses = []
	
	if block_names is None:
		losses = all_losses 
	else:
		if not isinstance(block_names, list):
			block_names = [block_names]
		for loss in all_losses:
			included = False 
			for block_name in block_names:
				if '/'.join([net_name_scope, block_name]) in loss.name:
					included=True 
					break 
			if included:
				losses.append(loss)
	if add_to_collection:
		total_regularization_loss = tf.add_n(losses, name='regularization_loss')
		tf.add_to_collection(collection_name, total_regularization_loss)
	return losses

def get_trainable_variables_with_block_names(net_name_scope='InceptionV3_pruned', block_names=None):
	all_trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
	if block_names is None:
		return all_trainable_variables
	variables = [] 
	if not isinstance(block_names, list):
		block_names = [block_names]
	for variable in all_trainable_variables:
		included = False
		for block_name in block_names:
			if '/'.join([net_name_scope, block_name]) in variable.op.name:
				included = True 
				break 
		if included:
			variables.append(variable)

	return variables 

def get_model_variables_with_block_names(net_name_scope='InceptionV3_pruned', block_names=None):
	all_trainable_variables = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
	if block_names is None:
		return all_trainable_variables
	variables = [] 
	if not isinstance(block_names, list):
		block_names = [block_names]
	for variable in all_trainable_variables:
		included = False
		for block_name in block_names:
			if '/'.join([net_name_scope, block_name]) in variable.op.name:
				included = True 
				break 
		if included:
			variables.append(variable)

	return variables

def get_update_ops_with_block_names(net_name_scope='InceptionV3_pruned', block_names=None):
	all_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	if block_names is None:
		return all_update_ops
	update_ops = []
	if not isinstance(block_names, list):
		block_names = [block_names]
	for op in all_update_ops:
		included = False
		for block_name in block_names:
			if '/'.join([net_name_scope, block_name])  in op.name:
				included = True 
				break 
		if included:
			update_ops.append(op)
	return update_ops 


###################################
# Functions for pruning variable  #
###################################
block_name_scopes_dict={
	'Mixed_5b': {'Branch_1': ['Conv2d_0a_1x1', 'Conv2d_0b_5x5'],
				 'Branch_2': ['Conv2d_0a_1x1', 'Conv2d_0b_3x3', 'Conv2d_0c_3x3'],
				 },
	'Mixed_5c': {'Branch_1': ['Conv2d_0b_1x1', 'Conv_1_0c_5x5'],
				 'Branch_2': ['Conv2d_0a_1x1', 'Conv2d_0b_3x3', 'Conv2d_0c_3x3'],
				 },
	'Mixed_5d': {'Branch_1': ['Conv2d_0a_1x1', 'Conv2d_0b_5x5'],
				 'Branch_2': ['Conv2d_0a_1x1', 'Conv2d_0b_3x3', 'Conv2d_0c_3x3'],
				 },
	'Mixed_6a': {'Branch_1':['Conv2d_0a_1x1', 'Conv2d_0b_3x3', 'Conv2d_1a_1x1']
				},
	'Mixed_6b': {'Branch_1': ['Conv2d_0a_1x1', 'Conv2d_0b_1x7', 'Conv2d_0c_7x1'],
				 'Branch_2': ['Conv2d_0a_1x1', 'Conv2d_0b_7x1', 'Conv2d_0c_1x7', 'Conv2d_0d_7x1', 'Conv2d_0e_1x7']
				 },
	'Mixed_6c': {'Branch_1': ['Conv2d_0a_1x1', 'Conv2d_0b_1x7', 'Conv2d_0c_7x1'],
				 'Branch_2': ['Conv2d_0a_1x1', 'Conv2d_0b_7x1', 'Conv2d_0c_1x7', 'Conv2d_0d_7x1', 'Conv2d_0e_1x7']
				 },
	'Mixed_6d': {'Branch_1': ['Conv2d_0a_1x1', 'Conv2d_0b_1x7', 'Conv2d_0c_7x1'],
				 'Branch_2': ['Conv2d_0a_1x1', 'Conv2d_0b_7x1', 'Conv2d_0c_1x7', 'Conv2d_0d_7x1', 'Conv2d_0e_1x7']
				 },
	'Mixed_6e': {'Branch_1': ['Conv2d_0a_1x1', 'Conv2d_0b_1x7', 'Conv2d_0c_7x1'],
				 'Branch_2': ['Conv2d_0a_1x1', 'Conv2d_0b_7x1', 'Conv2d_0c_1x7', 'Conv2d_0d_7x1', 'Conv2d_0e_1x7']
				 },
	'Mixed_7a': {'Branch_0': ['Conv2d_0a_1x1', 'Conv2d_1a_3x3'],
				 'Branch_1': ['Conv2d_0a_1x1', 'Conv2d_0b_1x7', 'Conv2d_0c_7x1', 'Conv2d_1a_3x3']
				},
	'Mixed_7b': {'Branch_1': ['Conv2d_0a_1x1', ['Conv2d_0b_1x3', 'Conv2d_0b_3x1']],
				 'Branch_2': ['Conv2d_0a_1x1', 'Conv2d_0b_3x3', ['Conv2d_0c_1x3', 'Conv2d_0d_3x1']]
				},
	'Mixed_7c': {'Branch_1': ['Conv2d_0a_1x1', ['Conv2d_0b_1x3', 'Conv2d_0c_3x1']],
				 'Branch_2': ['Conv2d_0a_1x1', 'Conv2d_0b_3x3', ['Conv2d_0c_1x3', 'Conv2d_0d_3x1']]
				},

}

def get_pruned_kernel_matrix(sess, prune_info, net_name_scope='InceptionV3'):
	init_values = {}

	for block_name in valid_block_names:
		if block_name not in prune_info:
			continue 
		block_kp = prune_info[block_name]['kp']
		block_name_scopes = block_name_scopes_dict[block_name]
		for branch_name, kernel_names in block_name_scopes.items():
			for i in range(len(kernel_names)-1):
				kernel_name = kernel_names[i]
				next_kernel_names = kernel_names[i+1]

				# get prune scopes and shorten scopes 
				prune_scope = '/'.join([net_name_scope, block_name, branch_name, kernel_name])
				if not isinstance(next_kernel_names, list):
					next_kernel_names = [next_kernel_names]
				shorten_scopes = ['/'.join([net_name_scope, block_name, branch_name, kernel_name]) for kernel_name in next_kernel_names]

				# get variables within the prune scope and shorten scopes 
				variables_to_prune = get_model_variables_within_scopes([prune_scope])
				variables_to_shorten = get_model_variables_within_scopes(shorten_scopes)

				# find the weights kernel from variables to prune
				index = find_weights_index(variables_to_prune)
				if variables_to_prune[index].op.name in init_values:
					variables_value = [init_values[variable_to_prune.op.name] for variable_to_prune in variables_to_prune]
				else:
					variables_value = sess.run(variables_to_prune) 
				weights = variables_value[index]

				# calculate the initial values for need-to-change variables
				filter_indexes = get_pruned_filter_indexes(weights, block_kp)
				new_weights = np.delete(weights, filter_indexes, axis=3)
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
				indexes = find_all_weights_index(variables_to_shorten)
				variables_value = sess.run(variables_to_shorten)
				for i in range(len(variables_to_shorten)):
					if i in indexes:
						new_value = np.delete(variables_value[i], filter_indexes, axis=2)
						# print('HG: shorten layers:', variables_to_shorten[i].op.name, variables_value[i].shape, '-->' ,new_value.shape)
					else:
						new_value = variables_value[i]
					
					init_values[variables_to_shorten[i].op.name] = new_value
	return init_values 


if __name__ == '__main__':
	kept_percentage=[0.3]*len(valid_block_names)
	block_names = valid_block_names
	prune_info = kept_percentage_sequence_to_prune_info(kept_percentage, block_names)
	set_prune_info_inputs(prune_info, end_points=None)
	pprint(prune_info)


