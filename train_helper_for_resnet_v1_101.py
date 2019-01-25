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

valid_block_names = ['block1/unit_1', 'block1/unit_2', 'block1/unit_3', \
	'block2/unit_1', 'block2/unit_2', 'block2/unit_3', 'block2/unit_4', \
	'block3/unit_1', 'block3/unit_2', 'block3/unit_3', 'block3/unit_4', \
	'block3/unit_5', 'block3/unit_6', 'block3/unit_7', 'block3/unit_8', \
	'block3/unit_9', 'block3/unit_10', 'block3/unit_11', 'block3/unit_12', \
	'block3/unit_13', 'block3/unit_14', 'block3/unit_15', 'block3/unit_16', \
	'block3/unit_17', 'block3/unit_18', 'block3/unit_19', 'block3/unit_20', \
	'block3/unit_21', 'block3/unit_22', 'block3/unit_23', 'block4/unit_1', \
	'block4/unit_2', 'block4/unit_3']


num_units = 33


block_to_num_units={'block1': 3, 'block2': 4, 'block3': 23, 'block4': 3}


def calculate_valid_block_names():
	valid_block_names = [] 
	blocks = ['block1', 'block2', 'block3', 'block4']
	for block in blocks:
		units = ['unit_'+str(index) for index in range(1, block_to_num_units[block]+1)]
		valid_block_names.extend([block+'/'+unit for unit in units])
	return valid_block_names




def config_to_kept_percentage_sequence(config, block_names, kept_percentages):
	if len(config) != len(block_names):
		print('len(config)', len(config))
		print('len(block_names)', len(block_names))
		raise ValueError(' len(config)!= len(block_names)')
	return [kept_percentages[option] for option in config] 


def valid_block_name_to_indexes(block_name):
	block_index = int(block_name.split('/')[0][-1])
	unit_index = int(block_name.split('_')[-1])
	return block_index, unit_index 

def kept_percentage_sequence_to_prune_info(kept_percentage, block_names):
	# each block name is a key, the value is a dictionary contains two keys: inputs and kp 
	if not isinstance(kept_percentage, list):
		kept_percentage = [kept_percentage]*len(block_names)
	prune_info = {}
	for kp, block_name in zip(kept_percentage, block_names):
		block_index, unit_index = valid_block_name_to_indexes(block_name)
		# print(block_name, block_index, unit_index)
		if block_index not in prune_info:
			prune_info[block_index]={}
		if unit_index not in prune_info[block_index]:
			prune_info[block_index][unit_index]={'inputs': None, 'prune_layers':{1: kp, 2: kp} }
			# pprint(prune_info)
	return prune_info 


BOTTLENECK_NAME = 'bottleneck_v1'
def set_prune_info_inputs(prune_info, end_points=None):
	if end_points:
		net_name_scope = end_points.keys()[0].split('/')[0]
	else:
		net_name_scope = 'resnet_v1_101'

	for block_index, block_info in prune_info.items():
		for unit_index, unit_info in block_info.items():
			if block_index == 1 and unit_index ==1:
				inputs_key = '/'.join([net_name_scope,'pool1'])
			else:
				block_name = 'block'+str(block_index)+'/unit_'+str(unit_index)
				inputs_key = '/'.join([net_name_scope, valid_block_names[valid_block_names.index(block_name)-1], BOTTLENECK_NAME])
			if end_points:
				unit_info['inputs'] = end_points[inputs_key]  
			else:
				unit_info['inputs'] = inputs_key

def valid_block_name_to_end_points_key(block_name, net_name_scope):
	inputs_key = '/'.join([net_name_scope, block_name, BOTTLENECK_NAME])
	return inputs_key 

###################################
# Qeury variables based on scopes #
###################################
# since variables names starts with the string InceptionV3, but the regularization loss 
# and update op starts with two string InceptionV3. The get variables or operations function
# are rewritten for InceptionV3 using keyword matching instead of scope search. 

def get_regularization_losses_with_block_names(net_name_scope='resnet_v1_101_pruned', 
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

def get_trainable_variables_with_block_names(net_name_scope='resnet_v1_101_pruned', block_names=None):
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

def get_model_variables_with_block_names(net_name_scope='resnet_v1_101_pruned', block_names=None):
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

def get_update_ops_with_block_names(net_name_scope='resnet_v1_101_pruned', block_names=None):
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
kernel_names = ['conv1', 'conv2', 'conv3']
BOTTLENECK_NAME = 'bottleneck_v1'
def get_pruned_kernel_matrix(sess, prune_info, net_name_scope='resnet_v1_101_pruned'):
	init_values = {}

	for block_name in valid_block_names:
		block_index, unit_index = valid_block_name_to_indexes(block_name)
		if block_index not in prune_info or (unit_index not in prune_info[block_index]):
			continue 
		block_kp = prune_info[block_index][unit_index]['prune_layers'][1]
		
		for i in range(len(kernel_names)-1):
			kernel_name = kernel_names[i]
			next_kernel_names = kernel_names[i+1]

			# get prune scopes and shorten scopes 
			prune_scope = '/'.join([net_name_scope, block_name, BOTTLENECK_NAME, kernel_name])
			if not isinstance(next_kernel_names, list):
				next_kernel_names = [next_kernel_names]
			shorten_scopes = ['/'.join([net_name_scope, block_name, BOTTLENECK_NAME, kernel_name]) for kernel_name in next_kernel_names]

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

	# test valid blocks names 
	# valid_block_names = calculate_valid_block_names()
	# print(valid_block_names)



