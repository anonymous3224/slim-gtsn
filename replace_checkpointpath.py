# replace some string from checkpoint path. 
# when move a model folder from titan to jupiter, the checkpoint path has to be changed for model restore. 
#


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys, re, os 


if len(sys.argv)==1:
	print('Please input the checkpoint file path')

filename = sys.argv[1]

old_str = "/lustre/atlas/proj-shared/csc160"
new_str = "/home/huiguan/tensorflow"

computer_name = os.environ['COMPUTER_NAME']
print('current computer name:', computer_name)
if computer_name=='titan':
	tmp = old_str
	old_str=new_str
	new_str= tmp

print('change', old_str, 'to', new_str)

new_lines = [] 
with open(filename, 'r') as f:
	lines = f.readlines()
	for line in lines:
		new_line = re.sub(old_str, new_str, line)
		new_lines.append(new_line)

with open(filename, 'w') as f:
	f.write(''.join(new_lines))



