import sys
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


filename=sys.argv[1]
tensor_name=''
all_tensors=False
print_tensors_in_checkpoint_file(file_name=filename, tensor_name=tensor_name, all_tensors=all_tensors)

