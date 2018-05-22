import argparse
import pdb
from .binvox_rw import read_as_3d_array, write
import numpy as np
parser = argparse.ArgumentParser()

# parser = ArgumentParser(description='redump_obj')
parser.add_argument('--file1', type=str, default=None)
parser.add_argument('--file2', type=str, default=None)
parser.add_argument('--union_file', type=str, default='union.binvox')
parser.add_argument('--inter_file', type=str, default='inter.binvox')


args = parser.parse_args()





file1_fp = open(args.file1,'rb')
file2_fp = open(args.file2,'rb')


voxel1 = read_as_3d_array(file1_fp)
voxel2 = read_as_3d_array(file2_fp)

voxel_union = voxel1.clone()
voxel_union.data = np.logical_or(voxel1.data, voxel2.data)


voxel_inter = voxel1.clone()
voxel_inter.data = np.logical_and(voxel1.data, voxel2.data)

union_file = open(args.union_file, 'w')
inter_file = open(args.inter_file, 'w')

write(voxel_union, union_file)
write(voxel_inter, inter_file)
union_file.close()
inter_file.close()

pdb.set_trace()