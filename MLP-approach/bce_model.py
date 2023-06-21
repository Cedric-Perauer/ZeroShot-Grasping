import torch
import torch.nn as nn

from dinov2.models.vision_transformer import vit_small, vit_large

class BCEGraspTransformer(nn.Module):

    def __init__(self, img_size=224, input_dim=384, output_dim=16, int_dim=32):
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
        self.patch_size = 14
        self.linear_head = nn.Sequential(
            nn.Linear(input_dim, int_dim),
            nn.ReLU(),
        )
        self.linear_head2 = nn.Sequential(
            nn.Linear(int_dim+2, output_dim),
            nn.ReLU(),
        )
        self.class_head =nn.Sequential(
            nn.Linear((output_dim*2)+2, 1),
            nn.Sigmoid()
        )
    def return_dino_features(self, img):
        return self.dinov2d_backbone.forward_features(img)['x_norm_patchtokens']

    def forward(self, feat1, feat2, p1, p2):
        f_reduce1 = torch.cat([self.linear_head(feat1), p1], dim=-1)
        f_reduce2 = torch.cat([self.linear_head(feat2), p2], dim=-1)
        conc_head = torch.cat([f_reduce1, f_reduce2], dim=-1)
        conc_head = torch.cat([conc_head, torch.abs(p1-p2)], dim=-1)
        return self.class_head(conc_head)


