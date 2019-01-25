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


# def prune_scopes_overlaped_with_shorten_scopes(prune_scopes, shorten_scopes):
# 	if len(set(prune_scopes) & set(shorten_scopes))>0:
# 		return True
# 	else:
# 		return False 


def get_pruned_kernel_matrix(sess, prune_scopes, shorten_scopes, kept_percentages=0.5):
	''' get the init values for pruned kernels in a greedy way'''
	init_values = {}

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
	return init_values 




###################################
# Functions for Pruning Scopes  #
###################################

def _get_block_name(block_index):
	""" 
	input: block index in string format. e.g. '1'
	output: block name in string formt	
	"""
	if int(block_index) not in {1, 2, 3, 4}:
		raise Exception('input is illegal: block_index='+ str(block_index))
	return 'block'+str(block_index)

def _get_unit_name(block_index, unit_index):
	block_index = int(block_index)
	unit_index = int(unit_index)
	if block_index==1 and unit_index in {1,2,3}:
		return 'unit_'+str(unit_index)
	elif block_index==2 and unit_index in {1, 2, 3, 4}:
		return 'unit_'+str(unit_index)
	elif block_index==3 and unit_index in {1, 2, 3, 4, 5, 6}:
		return 'unit_'+str(unit_index)
	elif block_index==4 and unit_index in {1,2,3}:
		return 'unit_'+str(unit_index)
	else:
		raise Exception('input is illegal: block_index='+ str(block_index) + ", unit_index=" + str(unit_index))

def _get_conv_name(conv_index):
	"""only prune the first two convolutional layers for one bottleneck"""
	conv_index = int(conv_index)
	if conv_index not in {1, 2, 3}:
		raise Exception('input is illegal: conv_index='+ str(conv_index))
	return 'bottleneck_v1/conv'+str(conv_index)

def get_block_unit_conv_name(s):
	if len(s)!=3:
		raise Exception('input is illegal: s='+s)
	return _get_block_name(s[0])+'/'+_get_unit_name(s[0], s[1])+'/'+_get_conv_name(s[-1])


def calculate_valid_indexed_prune_scopes():
	valid_indexes = set([])
	valid_conv_indexes = {1, 2}
	valid_unit_indexes = {1: {1,2,3}, 2:{1,2,3,4}, 3:{1,2,3,4,5,6}, 4:{1,2,3}}
	for block_index, unit_indexes in valid_unit_indexes.items():
		for unit_index in unit_indexes:
			for conv_index in valid_conv_indexes:
				index = "".join(map(str, [block_index, unit_index, conv_index]))
				valid_indexes.add(index)
	return valid_indexes 




valid_indexed_prune_scopes=['111', '112', '121', '122', '131', '132', \
	'211', '212', '221', '222', '231', '232', '241', '242', \
	'311', '312', '321', '322', '331', '332', '341', '342', '351', '352', '361', '362', \
	'411', '412', '421', '422', '431', '432']

valid_indexed_prune_scopes_for_units=[['111', '112'], ['121', '122'], ['131', '132'], \
	['211', '212'], ['221', '222'], ['231', '232'], ['241', '242'], \
	['311', '312'], ['321', '322'], ['331', '332'], ['341', '342'], ['351', '352'], ['361', '362'], \
	['411', '412'], ['421', '422'], ['431', '432']]

def indexed_prune_scopes_to_prune_info(indexed_prune_scopes, kept_percentage):
	prune_info = {}

	if not isinstance(kept_percentage, list):
		kept_percentage = [kept_percentage]*len(indexed_prune_scopes)
	# block indexes
	for i in range(len(indexed_prune_scopes)):
		prune_scope = indexed_prune_scopes[i] 
		if len(prune_scope)!=3:
			raise Exception('input is illegal: prune_scope='+ prune_scope)
		block_index = int(prune_scope[0])
		if block_index not in prune_info:
			prune_info[block_index] = {}
		unit_index = int(prune_scope[1])
		if unit_index not in prune_info[block_index]:
			prune_info[block_index][unit_index] = {'inputs': None, 'prune_layers':{}}
		conv_index = int(prune_scope[-1])
		prune_layers = prune_info[block_index][unit_index]['prune_layers']
		if conv_index not in prune_layers:
			prune_layers[conv_index] = kept_percentage[i]
	return prune_info 




# prune_unit_index_to_scope_map={
# 	'11': 'pool1',
# 	'12': 'block1/unit_1/bottleneck_v1',
# 	'13': 'block1/unit_2/bottleneck_v1',
# 	'21': 'block1/unit_3/bottleneck_v1',
# }
def _get_prune_units_inputs_scope(block_index, unit_index, net_name_scope):
	""" For local training, get the inputs name scope for the pruned units.
	block_index, unit_index is int"""
	if block_index==1:
		if unit_index==1:
			return net_name_scope+'/pool1'
		else:
			return net_name_scope+'/block1/unit_'+str(unit_index-1)+'/bottleneck_v1'
	elif block_index==2:
		if unit_index==1:
			return net_name_scope+'/block1/unit_3/bottleneck_v1'
		else:
			return net_name_scope+'/block2/unit_'+str(unit_index-1)+'/bottleneck_v1'
	elif block_index ==3:
		if unit_index==1:
			return net_name_scope+'/block2/unit_4/bottleneck_v1'
		else:
			return net_name_scope+'/block3/unit_'+str(unit_index-1)+'/bottleneck_v1'
	elif block_index ==4:
		if unit_index==1:
			return net_name_scope+'/block3/unit_6/bottleneck_v1'
		else:
			return net_name_scope+'/block4/unit_'+str(unit_index-1)+'/bottleneck_v1'
	else:
		raise ValueError('block_index=%d, unit_index=%d are illegal')


def _get_building_block_id(block_index, unit_index):
	""" the building block id starts with 0"""
	const = ['11', '12', '13', '21', '22', '23', '24', '31', '32', '33', '34', '35', '36', '41', '42', '43']
	s = str(block_index)+str(unit_index)
	return const.index(s)

def set_prune_units_inputs(prune_info, net_name_scope, end_points=None, block_size=1, building_block_ids=None):
	"""For local training: set the inputs value for prune_info. 
	building_block_ids specify the building block ids that requires an inputs. Otherwise, no inputs need to be set"""
	for block_index, block_dict in prune_info.items():
		for unit_index, unit_dict in block_dict.items():
			building_block_id = _get_building_block_id(block_index, unit_index)
			if building_block_ids is not None and building_block_id not in building_block_ids:
				continue 
			if building_block_id%block_size!=0:
				continue 
			inputs_scope = _get_prune_units_inputs_scope(block_index, unit_index, net_name_scope)
			if end_points==None:
				unit_dict['inputs'] = inputs_scope
			elif inputs_scope not in end_points:
				raise ValueError('inputs_scope not in end_points: inputs_scope='+inputs_scope)
			else:
				unit_dict['inputs'] = end_points[inputs_scope]


def get_prune_units_outputs_scope(indexed_prune_scope, net_name_scope='resnet_v1_50'):
	block_index = indexed_prune_scope[0]
	unit_index = indexed_prune_scope[1]
	return net_name_scope+'/block'+block_index+'/unit_'+unit_index+'/bottleneck_v1'

def get_prune_units_outputs_scopes(prune_info, net_name_scope='resnet_v1_50', block_size=1, building_block_ids=None):
	scopes = []
	for block_index in sorted(prune_info.keys()):
		block_dict = prune_info[block_index]
		for unit_index in sorted(block_dict.keys()):
			building_block_id = _get_building_block_id(block_index, unit_index)
			if building_block_ids is not None and building_block_id not in building_block_ids:
				continue 
			if (building_block_id+1)%block_size!=0:
				continue 
			indexed_prune_scope = str(block_index)+str(unit_index)
			scopes.append(get_prune_units_outputs_scope(indexed_prune_scope, net_name_scope))
	return scopes 


def get_prune_units_inputs_scope(indexed_prune_scope, net_name_scope='resnet_v1_50'):
	block_index = int(indexed_prune_scope[0])
	unit_index = int(indexed_prune_scope[1])
	return _get_prune_units_inputs_scope(block_index, unit_index, net_name_scope)
def get_prune_units_inputs_scopes(prune_info, net_name_scope='resnet_v1_50'):
	scopes = []
	for block_index in sorted(prune_info.keys()):
		block_dict = prune_info[block_index]
		for unit_index in sorted(block_dict.keys()):
			indexed_prune_scope = str(block_index)+str(unit_index)
			scopes.append(get_prune_units_inputs_scope(indexed_prune_scope, net_name_scope))
	return scopes 


def get_train_scope_for_local_train(indexed_prune_scope, net_name_scope='resnet_v1_50_pruned'):
	block_index = indexed_prune_scope[0]
	unit_index = indexed_prune_scope[1]
	return net_name_scope+'/block'+block_index+'/unit_'+unit_index+'/bottleneck_v1'

def get_train_scopes_for_local_train(prune_info, net_name_scope='resnet_v1_50_pruned', block_size=1):
	scopes = []
	block_scopes = [] 
	for block_index in sorted(prune_info.keys()):
		block_dict = prune_info[block_index]
		for unit_index in sorted(block_dict.keys()):
			indexed_prune_scope = str(block_index)+str(unit_index)
			block_scopes.append(get_train_scope_for_local_train(indexed_prune_scope, net_name_scope))
			building_block_id = _get_building_block_id(block_index, unit_index)
			if (building_block_id+1)%block_size==0:
				scopes.append(block_scopes)
				block_scopes=[]
	return scopes 



def indexed_prune_scopes_to_prune_scopes(indexed_prune_scopes, net_name_scope='resnet_v1_50'):
	# prune_scopes = ['resnet_v1_50/block1/unit_1/bottleneck_v1/conv1']
	prune_scopes = [get_block_unit_conv_name(scope) for scope in indexed_prune_scopes]
	prune_scopes = [net_name_scope+'/'+scope for scope in prune_scopes]
	return prune_scopes

def indexed_prune_scopes_to_shorten_scopes(indexed_prune_scopes, net_name_scope='resnet_v1_50'):
	# shorten_scopes = ['resnet_v1_50/block1/unit_1/bottleneck_v1/conv2']
	indexed_shorten_scopes = _indexed_prune_scopes_to_indexed_shorten_scopes(indexed_prune_scopes)
	return indexed_prune_scopes_to_prune_scopes(indexed_shorten_scopes, net_name_scope)


def _indexed_prune_scopes_to_indexed_shorten_scopes(indexed_prune_scopes):
	shorten_scopes = []
	for scope in indexed_prune_scopes:
		scope = int(scope)+1
		shorten_scopes.append(str(scope))
	return shorten_scopes

## helper functions for enumerating configurations and generate a subset of configs to evaluate. 
def config_to_indexed_prune_scopes(config, indexed_prune_scopes_for_units, kept_percentages):
    # print('HG: convert to indexed scopes from the config: %s' %(str(config)))
    indexed_prune_scopes = []
    prune_scopes_kept_percentages = [] 
    for i in xrange(len(config)):
        prune_option = config[i]
        kept_percentage = kept_percentages[prune_option]
        if kept_percentage==1.0:
            continue 
        scopes_for_unit = indexed_prune_scopes_for_units[i]
        indexed_prune_scopes.extend(scopes_for_unit)
        prune_scopes_kept_percentages.extend([kept_percentage]*len(scopes_for_unit))
    
    return indexed_prune_scopes, prune_scopes_kept_percentages
