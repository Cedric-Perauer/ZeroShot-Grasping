import torch
import torch.nn as nn

from dinov2.models.vision_transformer import vit_small, vit_base

class BCEGraspTransformer(nn.Module):

    def __init__(self, img_size=224, input_dim=768, output_dim=32, int_dim=256,input_cls=3):
        super(BCEGraspTransformer, self).__init__()
        self.img_size = img_size
        self.dinov2d_backbone = vit_base(
            img_size=518,
            patch_size=14,
            init_values=1.0e-05,
            ffn_layer="mlp",
            block_chunks=0,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
        )
        self.dinov2d_backbone.load_state_dict(torch.load('dinov2_vitb14_pretrain.pth'))
        for param in self.dinov2d_backbone.parameters():
            param.requires_grad = False
        self.patch_size = 14
        self.linear_head = nn.Sequential(
            nn.Linear(input_dim, int_dim),
            nn.ReLU(),
        )
        self.linear_head2 = nn.Sequential(
            nn.Linear(int_dim+1, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, 1),
            nn.Sigmoid()
        )
        
        self.linear_headvalid = nn.Sequential(
            nn.Linear(int_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, input_cls),
            nn.Sigmoid()
        )
        
        self.conv_head = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=256, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=True),
        )
        
        self.conv_linear_head = nn.Sequential(
            nn.Linear(256+1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward_dino_features(self, img):
        ret_list = self.dinov2d_backbone.forward_features(img)

        return ret_list['x_norm_patchtokens'], ret_list["x_norm_clstoken"]

    def forward_dino_attentions(self, img):
        return self.dinov2d_backbone.get_last_self_attention(img)


    def forward(self, feats, diffs):
        f_reduce = self.linear_head(feats)
        return self.linear_head2(torch.cat([f_reduce, diffs], dim=1))
    
    def forward_conv(self, feats):
        '''
        get conv forward features around a small feature patch 
        '''
        return self.conv_head(feats)
    
    def forward_both_convs(self,feats_fused,diffs):
        '''
        forward the stacked conv features through a FNN to get grasp output 
        '''
        return self.conv_linear_head(torch.cat([feats_fused, diffs], dim=1))
        
   
    
    def forward_valid(self, feats):
        f_reduce = self.linear_head(feats)
        return self.linear_headvalid(f_reduce)


