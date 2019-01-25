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

valid_block_names = ['Mixed_3b', 'Mixed_3c', \
					'Mixed_4a', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e', \
					'Mixed_5a', 'Mixed_5b', 'Mixed_5c']

num_units = 10

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



def set_prune_info_inputs(prune_info, end_points=None):
	for block_name, block_info in prune_info.items():
		if block_name == 'Mixed_3b':
			inputs_key = 'MaxPool_3a_3x3'
		else:
			inputs_key = valid_block_names[valid_block_names.index(block_name)-1]
		block_info['inputs'] = inputs_key if end_points is None else end_points[inputs_key]

###################################
# Qeury variables based on scopes #
###################################
# since variables names starts with the string InceptionV3, but the regularization loss 
# and update op starts with two string InceptionV3. The get variables or operations function
# are rewritten for InceptionV3 using keyword matching instead of scope search. 

def get_regularization_losses_with_block_names(net_name_scope='InceptionV2_pruned', 
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

def get_trainable_variables_with_block_names(net_name_scope='InceptionV2_pruned', block_names=None):
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

def get_update_ops_with_block_names(net_name_scope='InceptionV2_pruned', block_names=None):
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
	'Mixed_3b': {'Branch_1': ['Conv2d_0a_1x1', 'Conv2d_0b_3x3'],
				 'Branch_2': ['Conv2d_0a_1x1', 'Conv2d_0b_3x3', 'Conv2d_0c_3x3'],
				 },
	'Mixed_3c': {'Branch_1': ['Conv2d_0a_1x1', 'Conv2d_0b_3x3'],
				 'Branch_2': ['Conv2d_0a_1x1', 'Conv2d_0b_3x3', 'Conv2d_0c_3x3'],
				 },
	'Mixed_4a': {'Branch_0': ['Conv2d_0a_1x1', 'Conv2d_1a_3x3'],
				 'Branch_1': ['Conv2d_0a_1x1', 'Conv2d_0b_3x3', 'Conv2d_1a_3x3'],
				 },
	'Mixed_4b': {'Branch_1': ['Conv2d_0a_1x1', 'Conv2d_0b_3x3'],
				 'Branch_2': ['Conv2d_0a_1x1', 'Conv2d_0b_3x3', 'Conv2d_0c_3x3'],
				 },
	
	'Mixed_4c': {'Branch_1': ['Conv2d_0a_1x1', 'Conv2d_0b_3x3'],
				 'Branch_2': ['Conv2d_0a_1x1', 'Conv2d_0b_3x3', 'Conv2d_0c_3x3'],
				 },
	'Mixed_4d': {'Branch_1': ['Conv2d_0a_1x1', 'Conv2d_0b_3x3'],
				 'Branch_2': ['Conv2d_0a_1x1', 'Conv2d_0b_3x3', 'Conv2d_0c_3x3'],
				 },
	'Mixed_4e': {'Branch_1': ['Conv2d_0a_1x1', 'Conv2d_0b_3x3'],
				 'Branch_2': ['Conv2d_0a_1x1', 'Conv2d_0b_3x3', 'Conv2d_0c_3x3'],
				 },
	'Mixed_5a': {'Branch_0': ['Conv2d_0a_1x1', 'Conv2d_1a_3x3'],
				 'Branch_1': ['Conv2d_0a_1x1', 'Conv2d_0b_3x3', 'Conv2d_1a_3x3'],
				 },
	'Mixed_5b': {'Branch_1': ['Conv2d_0a_1x1', 'Conv2d_0b_3x3'],
				 'Branch_2': ['Conv2d_0a_1x1', 'Conv2d_0b_3x3', 'Conv2d_0c_3x3'],
				 },
	'Mixed_5c': {'Branch_1': ['Conv2d_0a_1x1', 'Conv2d_0b_3x3'],
				 'Branch_2': ['Conv2d_0a_1x1', 'Conv2d_0b_3x3', 'Conv2d_0c_3x3'],
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
				# print('HG: prune_scope', prune_scope)
				# print('HG: variables_to_prune')
				# pprint(variables_to_prune)
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


