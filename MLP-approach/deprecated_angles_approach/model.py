import torch 
import torch.nn as nn 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torchvision.models as models 
from dinov2.models.vision_transformer import vit_small, vit_large


class VisionLayer(nn.Module):
    def __init__(self, in_channels,nc,tokenW,tokenH):
        super(VisionLayer, self).__init__()
        self.in_channels = in_channels
        self.tokenW = tokenW
        self.tokenH = tokenH
        self.nc = nc 
        self.conv = nn.Conv2d(in_channels,nc,(1,1))
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = x.reshape(-1,self.in_channels,self.tokenH,self.tokenW)
        return self.relu(self.conv(x))

class GraspTransformer(nn.Module):
    
        def __init__(self,feature_layers=[11],angle_mode=False,img_size=224,SIM=False):
            super(GraspTransformer,self).__init__()
            '''
            angle_mode = True : uses the angle representation (x,y,theta_cos,theta_sin,w)
            angle_mode = False : uses the grasp point representation (xl,yl,xr,yr)
            '''
            self.img_size = img_size
            self.angle_mode = angle_mode
            self.feature_layers = feature_layers
            #vit14s had 11 layers max
            #self.dinov2d_backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.dinov2d_backbone = vit_small(
                    patch_size=14,
                    img_size=526,
                    init_values=1.0,
                    #ffn_layer="mlp",
                    block_chunks=0
            )
            self.dinov2d_backbone.load_state_dict(torch.load('dinov2_vits14_pretrain.pth'))
            ##freeze the dino layers
            self.nc = 16
            self.patch_size = 14
            self.pca = PCA(n_components=3)
            self.tokenw = int(self.img_size/14.)
            self.tokenh = int(self.img_size/14.)
            self.img_multiplier = int(self.img_size/224.)
            self.vision_layer_naive = VisionLayer(int(384 *len(self.feature_layers)),self.nc,self.tokenw,self.tokenh)
            self.vision_layer_similarity = VisionLayer(self.tokenh*self.tokenw,self.nc,self.tokenw,self.tokenh)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.pca_mode = False
            ##freeze dino layers
            for param in self.dinov2d_backbone.parameters(): 
                param.requires_grad = False
            
            self.out_params = 5 if self.angle_mode else 4
            self.resnet = models.resnet18(pretrained=True)
            #self.resnet50 = models.resnet50(pretrained=True)
            self.resnet.fc = nn.Identity()
            #self.resnet50.fc = nn.Identity()
            
            
            self.conv1 = VisionLayer(384,3,self.tokenw,self.tokenh)
            self.convC = VisionLayer(6,3,self.img_size,self.img_size)
            
            if SIM == False : 
                self.input_dim_naive = self.out_params + self.tokenw * self.tokenh * self.nc * 2
            else : 
                self.input_dim_similarity = self.out_params  + self.tokenh * self.tokenh * self.nc 
            
            self.mlp_head_resnet = nn.Sequential(
                        nn.Linear(512*2+self.out_params,128),
                        nn.ReLU(),
                        nn.Linear(128, self.out_params)
                    )
            
            
            if SIM == False : 
                self.mlp_head_naive = nn.Sequential(
                        nn.Linear(self.input_dim_naive,1024),
                        nn.ReLU(),
                        nn.Linear(1024, self.out_params)
                    )
            else :     
                self.mlp_head_similarity = nn.Sequential(
                            nn.Linear(self.input_dim_similarity,1024),
                            nn.ReLU(),
                            nn.Linear(1024, self.out_params)
                            )
            
            
        def process_query_label(self,label_query):
            pass #tbd grasp transformation of the query label into GK Net representation 
        
        def select_best_views(self,x_query,x_ref):
            '''
            input the different features and choose the best query image based on cosine similarity 
            '''
            features_query = self.dinov2d_backbone.get_intermediate_layers(x=x_query,n=self.feature_layers) 
            features_ref = self.dinov2d_backbone.get_intermediate_layers(x=x_ref,n=self.feature_layers) 
            features_query = torch.cat([feats for feats in features_query],dim=-1)
            features_ref = torch.cat([feats for feats in features_ref],dim=-1)
            #get cosine similarity 
            ref_features = features_ref[0].reshape(1,-1)
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            #import pdb; pdb.set_trace()
            query_features = features_query.reshape(features_query.shape[0],-1)
            #import pdb; pdb.set_trace()
            output = cos(ref_features,query_features)
            pred = torch.argmax(output)
            return pred, features_query[pred], features_ref[0] 
        
        def forward_inference(self,feats_ref,feats_query,query_label=None):
            '''
            Input the ref and query image and the query grasping labels
            1) Get dinov2 feature encoding of query and ref
            2) Concat features with query_label and pass it through an MLP to get the 
                output grasping label in the ref view
            query_label : [x,y,theta,w] GKNet encoding
            '''
            #query = torch.randn((1,4))
            #import pdb; pdb.set_trace()
            feats_ref = self.vision_layer(feats_ref).flatten().unsqueeze(0)
            feats_query = self.vision_layer(feats_query).flatten().unsqueeze(0)
            #import pdb; pdb.set_trace()
            ml_input = torch.cat([feats_ref,feats_query,query_label],dim=-1)
            mlp_forward = self.mlp_head(ml_input)
            x,y,theta,w = mlp_forward[:,0], mlp_forward[:,1], mlp_forward[:,2], mlp_forward[:,3]
            return x,y,theta,w
            
        def forward_naive(self,img, img_augmented,grasp_label):     
            '''
            forward image and augmented image through the dinov2 backbone and fuse it with the grasp label 
            this feature vector is then feed through an MLP to get the grasp position in the augmented image 
            '''
            img_feats_raw = self.dinov2d_backbone.forward_features(img)['x_norm_patchtokens']
            augmented_feats_raw = self.dinov2d_backbone.forward_features(img_augmented)['x_norm_patchtokens']
            
            
            img_feats = self.vision_layer_naive(img_feats_raw).reshape(img.shape[0],-1)
            augmented_feats = self.vision_layer_naive(augmented_feats_raw).reshape(img.shape[0],-1)
            
            
            ml_input = torch.cat([img_feats,augmented_feats,grasp_label],dim=-1)
            mlp_forward = self.mlp_head_naive(ml_input)
            if self.angle_mode == True : 
                center,theta_cos,theta_sin,w = mlp_forward[:,:2],  mlp_forward[:,2], mlp_forward[:,3],mlp_forward[:,4]
                center = nn.Sigmoid()(center)
                #theta_cos = nn.Tanh()(theta_cos)
                #theta_sin = nn.Tanh()(theta_sin)
                w = nn.Sigmoid()(w)
                return center,theta_cos,theta_sin,w, img_feats_raw, augmented_feats_raw
            else : 
                point_left, point_right = mlp_forward[:,:2], mlp_forward[:,2:]
                #point_left,point_right = nn.Sigmoid()(point_left), nn.Sigmoid()(point_right)
                return point_left, point_right, img_feats_raw, augmented_feats_raw 
        
        def forward_pca(self,img, img_augmented,grasp_label):     
            '''
            forward image and augmented image through the dinov2 backbone and fuse it with the grasp label 
            this feature vector is then feed through an MLP to get the grasp position in the augmented image 
            '''
            img_feats_raw = self.dinov2d_backbone.forward_features(img)['x_norm_patchtokens']
            augmented_feats_raw = self.dinov2d_backbone.forward_features(img_augmented)['x_norm_patchtokens']
            
            if self.pca_mode == True :
                self.pca.fit(img_feats_raw[0].cpu())
                self.pca.fit(augmented_feats_raw[0].cpu())
                pca_features_augmented = self.pca.transform(augmented_feats_raw[0].cpu())
                pca_features_augmented[:, 0] = (pca_features_augmented[:, 0] - pca_features_augmented[:, 0].min()) / \
                        (pca_features_augmented[:, 0].max() - pca_features_augmented[:, 0].min())
                        
                pca_features = self.pca.transform(img_feats_raw[0].cpu())
                pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()) / \
                        (pca_features[:, 0].max() - pca_features[:, 0].min())
                    
                im_dim = int(pca_features.shape[0] ** 0.5)
                pca_features = pca_features.reshape(1,3,im_dim,im_dim) 
                pca_features_augmented = pca_features_augmented.reshape(1,3,im_dim,im_dim)
                
                pca_features = torch.from_numpy(pca_features).to(self.device).to(torch.float32)
                pca_features_augmented = torch.from_numpy(pca_features_augmented).to(self.device).to(torch.float32)
                pca_features = self.resnet(pca_features)
                pca_features_augmented = self.resnet(pca_features_augmented)
            
            else : 
                im_dim = int(img_feats_raw.shape[1] ** 0.5)
                img_feats_raw_re = img_feats_raw.reshape(img_feats_raw.shape[0],384,im_dim,im_dim)
                augmented_feats_raw_re = augmented_feats_raw.reshape(img_feats_raw.shape[0],384,im_dim,im_dim)
                pca_features = self.resnet(self.conv1(img_feats_raw_re))
                pca_features_augmented = self.resnet(self.conv1(augmented_feats_raw_re)) 
            
            ml_input = torch.cat([pca_features,pca_features_augmented,grasp_label],dim=-1)
            mlp_forward = self.mlp_head_resnet(ml_input)
            if self.angle_mode == True : 
                center,theta_cos,theta_sin,w = mlp_forward[:,:2],  mlp_forward[:,2], mlp_forward[:,3],mlp_forward[:,4]
                center = nn.Sigmoid()(center)
                #theta_cos = nn.Tanh()(theta_cos)
                #theta_sin = nn.Tanh()(theta_sin)
                w = nn.Sigmoid()(w)
                return center,theta_cos,theta_sin,w, img_feats_raw, augmented_feats_raw
            else : 
                point_left, point_right = mlp_forward[:,:2], mlp_forward[:,2:]
                #point_left,point_right = nn.Sigmoid()(point_left), nn.Sigmoid()(point_right)
                return point_left, point_right, img_feats_raw, augmented_feats_raw 
            
        def forward_attention(self,img, img_augmented,grasp_label):     
            '''
            forward image and augmented image through the dinov2 backbone and fuse it with the grasp label 
            this feature vector is then feed through an MLP to get the grasp position in the augmented image 
            '''
            
            attentions_img = self.dinov2d_backbone.get_last_self_attention(img)
            attentions_aug_img = self.dinov2d_backbone.get_last_self_attention(img_augmented)
            nh = attentions_img.shape[1] # number of head
            attentions_img = attentions_img[0, :, 0, 1:].reshape(nh, -1)
            attentions_aug_img = attentions_aug_img[0, :, 0, 1:].reshape(nh, -1)
            # weird: one pixel gets high attention over all heads?
            #attentions_img[:, 283] = 0 
            #attentions_aug_img[:,283] = 0
            
            w_featmap, h_featmap = img.shape[2] // self.patch_size, img.shape[3] // self.patch_size 
            
            attentions_img = attentions_img.reshape(nh, w_featmap, h_featmap)
            attentions_img = nn.functional.interpolate(attentions_img.unsqueeze(0), scale_factor=self.patch_size, mode="nearest")
            
            attentions_aug_img = attentions_aug_img.reshape(nh, w_featmap, h_featmap)
            attentions_aug_img = nn.functional.interpolate(attentions_aug_img.unsqueeze(0), scale_factor=self.patch_size, mode="nearest")
            
            img_features = self.resnet(self.convC(attentions_img))
            features_augmented = self.resnet(self.convC(attentions_aug_img))
            
            ml_input = torch.cat([img_features,features_augmented,grasp_label],dim=-1)
            mlp_forward = self.mlp_head_resnet(ml_input)
            
            #import pdb; pdb.set_trace()
            
            if self.angle_mode == True : 
                center,theta_cos,theta_sin,w = mlp_forward[:,:2],  mlp_forward[:,2], mlp_forward[:,3],mlp_forward[:,4]
                center = nn.Sigmoid()(center)
                #theta_cos = nn.Tanh()(theta_cos)
                #theta_sin = nn.Tanh()(theta_sin)
                w = nn.Sigmoid()(w)
                return center,theta_cos,theta_sin,w, attentions_img, attentions_aug_img
            else : 
                point_left, point_right = mlp_forward[:,:2], mlp_forward[:,2:]
                #point_left,point_right = nn.Sigmoid()(point_left), nn.Sigmoid()(point_right)
                return point_left, point_right, attentions_img, attentions_aug_img 
            
        def forward_similarity(self,img, img_augmented,grasp_label):     
            '''
            forward image and augmented image through the dinov2 backbone and fuse it with the grasp label 
            this feature vector is then feed through an MLP to get the grasp position in the augmented image 
            '''
            img_feats_raw = self.dinov2d_backbone.forward_features(img)['x_norm_patchtokens']
            #import pdb; pdb.set_trace()
            augmented_feats_raw = self.dinov2d_backbone.forward_features(img_augmented)['x_norm_patchtokens']
            
            similarity = torch.bmm(augmented_feats_raw,torch.transpose(img_feats_raw,1,2))
            reduced_similarity = self.vision_layer_similarity(similarity).reshape(img.shape[0],-1)
            #import pdb; pdb.set_trace()
            ml_input = torch.cat([reduced_similarity,grasp_label],dim=-1)
            mlp_forward = self.mlp_head_similarity(ml_input)
            if self.angle_mode == True : 
                center,theta_cos,theta_sin,w = mlp_forward[:,:2],  mlp_forward[:,2], mlp_forward[:,3],mlp_forward[:,4]
                center = nn.Sigmoid()(center)
                #theta_cos = nn.Tanh()(theta_cos)
                #theta_sin = nn.Tanh()(theta_sin)
                w = nn.Sigmoid()(w)
                return center,theta_cos,theta_sin,w, img_feats_raw, augmented_feats_raw
            else : 
                point_left, point_right = mlp_forward[:,:2], mlp_forward[:,2:]
                #point_left,point_right = nn.Sigmoid()(point_left), nn.Sigmoid()(point_right)
                return point_left, point_right , img_feats_raw, augmented_feats_raw
            
            
            
            
            
            
if __name__ == '__main__':
    model = GraspTransformer()
    query_label = torch.randn((1,4))
    x_ref, x_query = torch.randn((1,3,224,224)), torch.randn((1,3,224,224))
    out = model(x_ref, x_query,query_label) 