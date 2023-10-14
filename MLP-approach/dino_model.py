import torch
import torch.nn as nn

from dinov2.models.vision_transformer import vit_small, vit_base
from torchvision import models 

class DinoModel(nn.Module):

    def __init__(self):
        super(DinoModel, self).__init__()
        #self.img_size = img_size
        self.dinov2d_backbone = vit_small(
            img_size=518,
            patch_size=14,
            init_values=1.0e-05,
            ffn_layer="mlp",
            block_chunks=0,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
        )
        
        self.dinov2d_backbone.load_state_dict(torch.load('dinov2_vits14_pretrain.pth'))
        for param in self.dinov2d_backbone.parameters():
            param.requires_grad = False
            
    def forward(self, img):
        ret_list = self.dinov2d_backbone.forward_features(img)

        return ret_list['x_norm_patchtokens'], ret_list["x_norm_clstoken"]