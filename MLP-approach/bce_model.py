import torch
import torch.nn as nn

from dinov2.models.vision_transformer import vit_small, vit_large

class BCEGraspTransformer(nn.Module):

    def __init__(self, img_size=224, input_dim=384, output_dim=32, int_dim=128):
        super(BCEGraspTransformer, self).__init__()
        self.img_size = img_size
        self.dinov2d_backbone = vit_small(
            patch_size=14,
            img_size=526,
            init_values=1.0,
            # ffn_layer="mlp",
            block_chunks=0
        )
        self.dinov2d_backbone.load_state_dict(torch.load('dinov2_vits14_pretrain.pth'))
        for param in self.dinov2d_backbone.parameters():
            param.requires_grad = False
        self.patch_size = 14
        self.linear_head = nn.Sequential(
            nn.Linear(input_dim, int_dim),
            nn.ReLU(),
        )
        self.linear_head2 = nn.Sequential(
            nn.Linear(int_dim+4, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, 1),
            nn.Sigmoid()
        )
        
        self.linear_headvalid = nn.Sequential(
            nn.Linear(int_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, 1),
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
    
    def forward_valid(self, feats):
        f_reduce = self.linear_head(feats)
        return self.linear_headvalid(f_reduce)


