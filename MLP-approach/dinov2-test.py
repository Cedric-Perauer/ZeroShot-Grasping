import torch 
import torch.nn as nn 

class GraspTransformer(nn.Module):
    
        def __init__(self,feature_layers=[9,11]):
            super(GraspTransformer,self).__init__()
            
            self.feature_layers = feature_layers
            #vit14s had 11 layers max
            self.dinov2d_backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            
            
        def process_query_label(self,label_query):
            pass #tbd grasp transformation of the query label into GK Net representation 
        
        def select_best_views(self,features_queries,x_ref):
            '''
            input the different features and choose the best query image based on cosine similarity 
            '''
            pass 
        
        def forward(self,x_ref,x_query,query_label=None):
            '''
            Input the ref and query image and the query grasping labels
            1) Get dinov2 feature encoding of query and ref
            2) Concat features with query_label and pass it through an MLP to get the 
                output grasping label in the ref view
            query_label : [x,y,theta,w] GKNet encoding
            '''
            features_query = self.dinov2d_backbone.get_intermediate_layers(x=x_query,n=self.feature_layers) 
            features_ref = self.dinov2d_backbone.get_intermediate_layers(x=x_ref,n=self.feature_layers) 
            features_query = torch.cat([feats for feats in features_query],dim=-1).flatten()
            features_ref = torch.cat([feats for feats in features_ref],dim=-1).flatten()
            query_label = query_label.flatten()
            feature_fusion = torch.cat([features_ref,features_query,query_label])
            import pdb; pdb.set_trace()
            
            
                        
model = GraspTransformer()
query_label = torch.randn((1,4))
x_ref, x_query = torch.randn((1,3,224,224)), torch.randn((1,3,224,224))
out = model(x_ref, x_query,query_label)

        
