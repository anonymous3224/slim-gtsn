# use this script to generate the configurations offline
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import cPickle as pickle 
import numpy as np 
import random 
import os 

def get_special_configurations(num_units, num_options):
    '''get a set of special configurations. e.g.: (0, 0, 0), (1, 1, 1)'''
    special_configs = [list([x]*num_units) for x in range(num_options)]
    return special_configs


# picke load and save 
def check_directory(path):
  if not os.path.isdir(path):
    os.makedirs(path)

def save_pickle(save_path, save_name, save_object):
  check_directory(save_path)
  filepath = os.path.join(save_path, save_name)
  pickle.dump(save_object, open(filepath,"wb" ))
  print('File saved to:', filepath)

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

def generate_sampled_configurations(num_units, num_options, n):
  # first include some special configs
  configs = get_special_configurations(num_units, num_options)
  if len(configs)>=n:
    return configs[:n]
  # then sample some extra configs based on close-to uniform distribution 
  upbound = _get_num_samples_per_sum_upbound(num_units, num_options, n-len(configs))
  print('num_samples_per_sum upbound:', upbound)

  # use configs_dict and sum_cnt_dict to track the number of samples
  cnt_dict = {}
  for item in configs:
    sum_value = np.sum(item)
    if sum_value not in cnt_dict:
      cnt_dict[sum_value] = 0
    cnt_dict[sum_value] +=1

  # generate random samples 
  random.seed(1992)
  total_cnts = 0 
  while len(configs)<n:
    config = []
    for i in range(num_units):
      config.append(random.randint(0, num_options-1))
    total_cnts+=1
    if config not in configs:
      sum_value = np.sum(config)
      # if the value doesn't exist before
      if sum_value not in cnt_dict:
        configs.append(config)
        cnt_dict[sum_value]=1
      # check if the cnt reaches the upbound
      elif cnt_dict[sum_value]< upbound:
        configs.append(config)
        cnt_dict[sum_value] +=1 

  print('total_cnts=', total_cnts)
  print(cnt_dict)
  return configs 


def _get_num_samples_per_sum_upbound(num_units, num_options, n):
  max_sum_value = num_units * (num_options -1)
  unique_number_of_sum = max_sum_value + 1 
  num_samples_per_sum = int(n/unique_number_of_sum)
  return 1.5* num_samples_per_sum 


if __name__ == '__main__':
	num_units = 15
	num_options = 3
	num_configs = 500 
	 
	configs = generate_sampled_configurations(num_units, num_options, num_configs)
	print_list('configs', configs)
	save_path = './configs'
	save_name = 'configs_%d_units_%d_options_%d.p' %(num_configs, num_units, num_options)
	save_pickle(save_path, save_name, save_object=configs)


