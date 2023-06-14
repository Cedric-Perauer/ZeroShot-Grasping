import torch 
import torch.nn as nn 


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
    
        def __init__(self,feature_layers=[11],angle_mode=False):
            super(GraspTransformer,self).__init__()
            '''
            angle_mode = True : uses the angle representation (x,y,theta_cos,theta_sin,w)
            angle_mode = False : uses the grasp point representation (xl,yl,xr,yr)
            '''
            self.angle_mode = angle_mode
            self.feature_layers = feature_layers
            #vit14s had 11 layers max
            self.dinov2d_backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            ##freeze the dino layers
            self.nc = 16
            self.tokenw = 16
            self.tokenh = 16 
            self.vision_layer_naive = VisionLayer(384*len(self.feature_layers),self.nc,self.tokenw,self.tokenh)
            self.vision_layer_similarity = VisionLayer(256,self.nc,self.tokenw,self.tokenh)
            ##freeze dino layers
            for param in self.dinov2d_backbone.parameters(): 
                param.requires_grad = False
            
            self.out_params = 5 if self.angle_mode else 4
            
            self.input_dim_naive = self.out_params + 2 * self.tokenw * self.tokenh * self.nc
            self.input_dim_similarity = self.out_params  + 16 * 16 * self.nc
            
            self.mlp_head_naive = nn.Sequential(
                    nn.Linear(self.input_dim_naive,128),
                    nn.ReLU(),
                    nn.Linear(128, self.out_params)
                    )
            
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
            img_feats = self.dinov2d_backbone.forward_features(img)['x_norm_patchtokens']
            augmented_feats = self.dinov2d_backbone.forward_features(img_augmented)['x_norm_patchtokens']
            img_feats = self.vision_layer(img_feats).reshape(img.shape[0],-1)
            augmented_feats = self.vision_layer_naive(augmented_feats).reshape(img.shape[0],-1)
            #import pdb; pdb.set_trace()
            ml_input = torch.cat([img_feats,augmented_feats,grasp_label],dim=-1)
            mlp_forward = self.mlp_head_naive(ml_input)
            if self.angle_mode == True : 
                center,theta_cos,theta_sin,w = mlp_forward[:,:2],  mlp_forward[:,2], mlp_forward[:,3],mlp_forward[:,4]
                center = nn.Sigmoid()(center)
                #theta_cos = nn.Tanh()(theta_cos)
                #theta_sin = nn.Tanh()(theta_sin)
                w = nn.Sigmoid()(w)
                return center,theta_cos,theta_sin,w
            else : 
                point_left, point_right = mlp_forward[:,:2], mlp_forward[:,2:]
                point_left,point_right = nn.Sigmoid()(point_left), nn.Sigmoid()(point_right)
                return point_left, point_right 
            
        def forward_similarity(self,img, img_augmented,grasp_label):     
            '''
            forward image and augmented image through the dinov2 backbone and fuse it with the grasp label 
            this feature vector is then feed through an MLP to get the grasp position in the augmented image 
            '''
            img_feats = self.dinov2d_backbone.forward_features(img)['x_norm_patchtokens']
            augmented_feats = self.dinov2d_backbone.forward_features(img_augmented)['x_norm_patchtokens']
            similarity = torch.bmm(augmented_feats,torch.transpose(img_feats,1,2))
            reduced_similarity = self.vision_layer_similarity(similarity).reshape(img.shape[0],-1)
            ml_input = torch.cat([reduced_similarity,grasp_label],dim=-1)
            mlp_forward = self.mlp_head_similarity(ml_input)
            if self.angle_mode == True : 
                center,theta_cos,theta_sin,w = mlp_forward[:,:2],  mlp_forward[:,2], mlp_forward[:,3],mlp_forward[:,4]
                center = nn.Sigmoid()(center)
                #theta_cos = nn.Tanh()(theta_cos)
                #theta_sin = nn.Tanh()(theta_sin)
                w = nn.Sigmoid()(w)
                return center,theta_cos,theta_sin,w
            else : 
                point_left, point_right = mlp_forward[:,:2], mlp_forward[:,2:]
                #point_left,point_right = nn.Sigmoid()(point_left), nn.Sigmoid()(point_right)
                return point_left, point_right 
            
            
            
            
            
            
if __name__ == '__main__':
    model = GraspTransformer()
    query_label = torch.randn((1,4))
    x_ref, x_query = torch.randn((1,3,224,224)), torch.randn((1,3,224,224))
    out = model(x_ref, x_query,query_label) 