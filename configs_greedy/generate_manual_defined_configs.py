# generate manually defined config 

import itertools 
from pprint import pprint 

num_blocks= [3, 4, 6, 3]
options = [[0.6, 0.7], [0.6, 0.7], [0.4, 0.5], [0.4, 0.5]]


combinations = list(itertools.product(*options))
print len(combinations) 

# extend combinations to config
configs = [] 
for comb in combinations:
	config = [] 
	for i, kp in enumerate(comb):
		config.extend([kp]*num_blocks[i])
	configs.append(config)

for config in configs:
	print config 

# save the configs into a file 
with open("manual_configs.txt", 'w') as f:
	for config in configs:
		f.write(", ".join(map(str, config))+'\n')
