import torch
import pdb

def weightsInit(m):
  name = str(type(m))
  if 'Conv3d' in name:
    m.weight.data = m.weight.data.normal_(mean=0,std=0.02)
    m.bias.data = m.bias.data.zero_()
  elif 'BatchNorm3d' in name:
    m.weight.data = m.weight.data.normal_(mean=1.0, std=0.02)
    m.bias.data = m.bias.data.zero_()