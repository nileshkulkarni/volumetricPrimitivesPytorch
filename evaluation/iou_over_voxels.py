## Assuming there are two directories.
## GT and Prediction, given as input.
## Filenames in both directories are same.
## Read both binvox files and compute IoU.
'''
python -m evaluation.iou_over_voxels --output_dir /nfs.yoda/nileshk/grass_binvox/chair_aligned_obj_obb_200/ --gt_dir /nfs.yoda/nileshk/grass_binvox/chair_aligned_obj_obb_200/ --result_file result.txt
'''
import glob
import os.path as osp
from .command_line import  get_args
args = get_args()
from .binvox_rw import read_as_3d_array
import numpy as np
import pdb
output_binvox_files = []

for filename in glob.glob(osp.join(args.output_dir, '*.binvox')):
    output_binvox_files.append(filename)

print("Evaluating on {} files".format(len(output_binvox_files)))


def compute_iou(voxel_1, voxel_2):
    intersection = np.logical_and(voxel_1.data, voxel_2.data)
    union = np.logical_or(voxel_1.data, voxel_2.data)
    iou = np.sum(intersection)/ np.sum(union)
    return iou


def evaluate_for(output_binvox_file, gt_binvox_file):
    output_fp = open(output_binvox_file,'rb')
    gt_fp = open(gt_binvox_file,'rb')
    output_voxels = read_as_3d_array(output_fp)
    gt_voxels = read_as_3d_array(gt_fp)

    iou = compute_iou(output_voxels, gt_voxels)
    return iou


result_file = open(args.result_file,'w')

ious = []
i = 0
for filename in output_binvox_files:
    filename = osp.basename(filename)
    gt_filename = osp.join(args.gt_dir, filename)
    output_filename = osp.join(args.output_dir, filename)
    if osp.isfile(output_filename) and osp.isfile(gt_filename):
        iou = evaluate_for(output_filename, gt_filename)
        result_file.write('{} : {}\n'.format(filename, iou))
        ious.append(iou)
        i = i +1

print('Results written to {}'.format(args.result_file))
print('Summary Results for {} files '.format(i))

ious = np.array(ious)

print('Mean : {} '.format(np.mean(ious)))
print('Median : {} '.format(np.median(ious)))





