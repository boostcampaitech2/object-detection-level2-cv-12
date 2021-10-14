import torch
import torch.nn as nn
import urllib.request
import PIL.Image as Image
import torchvision.transforms as transforms
import numpy as np
from concat_fpn_output import concat_feature_maps
from attention_layers import Scale_Aware_Layer, Spatial_Aware_Layer, Task_Aware_Layer
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

backbone_name = 'resnet50'
backbone = resnet_fpn_backbone(backbone_name=backbone_name, pretrained=True)

# Passing image through backbone
F = backbone(img_tensor)
print(type(F))

num_levels = 0
# F contains all the outputs of the Feature Pyramid Network
for level, level_tensor in F.items():
    print('Level: {} \t Shape: {}'.format(level, level_tensor.shape))
    try:
        if isinstance(int(level), int):
            num_levels+=1
    except ValueError:
        pass
print('{} total levels'.format(num_levels))



concat_layer = concat_feature_maps()
F = concat_layer(F)
print('Shape: {}'.format(F.shape))

L, S, C = F.shape[1:]
print('Median HeightxWidth: {}x{}'.format(int(np.sqrt(S)), int(np.sqrt(S))))
print("F Dimensions\nL: {} S: {} C: {}".format(L, S, C))

scale_layer = Scale_Aware_Layer(s_size=S)
scale_output = scale_layer(F)
print(f'Shape of scale output: {scale_output.shape}')

spatial_layer = Spatial_Aware_Layer(L_size=L)
spatial_output = spatial_layer(scale_output)
print(f'Shape of scale output: {spatial_output.shape}')

task_layer = Task_Aware_Layer(num_channels=C)
task_output = task_layer(spatial_output)
print(f'Shape of scale output: {task_output.shape}')
from DyHead import DyHead_Block

dyhead_block = DyHead_Block(L=L, S=S, C=C)
print(dyhead_block)
from DyHead import DyHead

num_blocks = 6 # This is the baseline given in the paper
full_head = DyHead(num_blocks, L, S, C)
print(full_head)