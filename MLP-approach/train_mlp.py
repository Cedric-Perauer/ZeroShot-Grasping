import os
import argparse

import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision 
from torchvision import transforms
from PIL import Image, ImageDraw
from matplotlib.patches import Polygon
import cv2
import math
import shutil
import einops
from einops import repeat
from dataset_reference import TestDataset
from dataset_augment import AugmentDataset
from model import GraspTransformer

LOG_DIR = 'logs/'
parser = argparse.ArgumentParser()

parser.add_argument('--co3d_root', type=str, default="", 
                    help='Root path to CO3D dataset')
parser.add_argument('--hub_dir', type=str, default="", 
                    help='Path to directory that pre-trained networks .pth are in')
parser.add_argument('--note', type=str, default="",
                    help='Note to flag experiment')
parser.add_argument('--log_dir', type=str, default=LOG_DIR, 
                    help='Path to log directory, under which exps get folders')
parser.add_argument('--best_frame_mode', type=str, default="corresponding_feats_similarity",
                    choices=['corresponding_feats_similarity', 'global_similarity',
                             'ref_to_target_saliency_map_iou', 'cylical_dists_to_saliency_map_iou'],
                    help='How to identify the best frame for correspondence')
parser.add_argument('--plot_results', action='store_true')
parser.add_argument('--num_workers', type=int, default=4,
                    help='How many workers in dataloader')
parser.add_argument('--num_plot_examples_per_batch', type=int, default=1,
                    help='How many plots to make per batch')
# Model parameters
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_samples_per_class', type=int, default=100, 
                    help='Number of examples to test on per category')
parser.add_argument('--num_correspondences', type=int, default=50, 
                    help='Number of correspondences to return from matching')
parser.add_argument('--n_target', type=int, default=4, 
                    help='Number of frames to draw as target/query set')
parser.add_argument('--patch_size', type=int, default=8, 
                    help='ViT model to use: which patch size')
parser.add_argument('--binning', type=str, default='log', choices=['log', 'gaussian', 'none'],
                    help='Method of combining/binning patch features into descriptors')
parser.add_argument('--kmeans', action='store_true',
                    help='Use K-means clustering step for higher diversity correspondences (boosts pose estimates)')
parser.add_argument('--ransac_thresh', type=float, default=0.2, 
                    help='Threshold for inliers in the ransac loop')
parser.add_argument('--take_best_view', action='store_true',
                    help='Rather than solving rigid body transform, just take "best" view as pose estimate')
parser.add_argument('--also_take_best_view', action='store_true',
                    help='Run both the full method, and the above method: log both results')

# From argparse
args = parser.parse_args()
# ---------------
# EXPERIMENT PARAMS
# ---------------
device = torch.device('cuda:0')
# ---------------
# SET UP DESCRIPTOR CLASS
# ---------------
image_norm_mean = (0.485, 0.456, 0.406)
image_norm_std = (0.229, 0.224, 0.225)
image_size = 224

def get_transform():
        image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_norm_mean, std=image_norm_std)
        ])
        return image_transform

def tile_ims_horizontal_highlight_best(ims, gap_px=20, highlight_idx=None):
    cumul_offsets = [0]
    for im in ims:
        cumul_offsets.append(cumul_offsets[-1]+im.width+gap_px)
    max_h = max([im.height for im in ims])
    dst = Image.new('RGB', (cumul_offsets[-1], max_h), (255, 255, 255))
    for i, im in enumerate(ims):
        dst.paste(im, (cumul_offsets[i], (max_h - im.height) // 2))
        
        if i == highlight_idx:
            img1 = ImageDraw.Draw(dst)  
            # shape is defined as [(x1,y1), (x2, y2)]
            shape = [(cumul_offsets[i],(max_h - im.height) // 2), 
                     (cumul_offsets[i]+im.width, max_h-(max_h - im.height) // 2)]
            img1.rectangle(shape, fill = None, outline ="green", width=6)

    return dst

def denorm_torch_to_pil(image):
        image = image * torch.Tensor(image_norm_std)[:, None, None]
        image = image + torch.Tensor(image_norm_mean)[:, None, None]
        return Image.fromarray((image.permute(1, 2, 0) * 255).numpy().astype(np.uint8))
    
def torch_to_pil(image):
        return (image.permute(1, 2, 0) * 255).numpy().astype(np.uint8)

def rotate_grasping_rectangle(grasp_points,H):
    '''
    inputs : 
        H : homography matrix to transform the boxes to the new frame
        grasp_points : array of all the grasp rectangles (4 each)
        
    '''
    grasp_points = grasp_points.astype(np.float32).reshape(1,-1,2)
    output = cv2.perspectiveTransform(grasp_points, H) 
    output = output.reshape(-1,4,2)
    
    return output
    

plot_results = args.plot_results


co3d_root = args.co3d_root
log_dir = args.log_dir
os.makedirs(log_dir, exist_ok=True)

pretrain_paths = {
    16: os.path.join(args.hub_dir, 'dino_vitbase16_pretrain.pth'),
    8: os.path.join(args.hub_dir, 'dino_deitsmall8_pretrain.pth'),
}


kmeans_str = 'Kmeans' if args.kmeans else 'NoKmeans'
note = args.note
log_dir = os.path.join(log_dir,
                       f"{args.num_correspondences}corr_{kmeans_str}_{args.binning}bin_{args.ransac_thresh}ransac_{args.n_target}tgt_{note}")

folder_name = 'crops/'


# ---------------
# LOAD MODEL
# ---------------
pretrain_path = pretrain_paths[args.patch_size]

model = GraspTransformer()
model = model.to(device)

categories = ["backpack", "bicycle", "book", "car", "chair", "hairdryer", "handbag",
              "hydrant", "keyboard", "laptop", "motorcycle", "mouse", "remote", 
              "teddybear", "toaster", "toilet", "toybus", "toyplane", "toytrain", "toytruck"]
idx_plot = 0

image_transform = get_transform()
CROP = False 
dataset = TestDataset(image_transform=image_transform,num_targets=args.n_target,vis=True,crop=CROP)
dataset = AugmentDataset(image_transform=image_transform,num_targets=args.n_target,vis=True,crop=CROP,overfit=True)
dataset.img_cnt = 0 

vis_dir = 'vis_out/'

os.makedirs(vis_dir, exist_ok=True)

optim = torch.optim.Adam(model.parameters(), lr=1e-3)
loss = torch.nn.MSELoss()

epochs = 100

for epoch in range(epochs) :
    
    dataloader = DataLoader(dataset, batch_size = args.batch_size,num_workers=0,shuffle=False)
    categories = dataset.classes
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        data = batch
        
        img = data['img']
        augmented_img = data['img_augmented']
        gknet_label = data['img_grasp']
        augmented_gknet_label = data['img_augmented_grasp']
        batch_size = img.shape[0]
        points_grasp = data['points_grasp']
        
        ##for the visualisation of the grasp 
        VIS = True
        if VIS : 
            for i in range(batch_size):
                w = augmented_gknet_label[i][4] * 224
                x,y = augmented_gknet_label[i][0] * 224 , augmented_gknet_label[i][1] * 224
                angle = math.atan2(augmented_gknet_label[i][3],augmented_gknet_label[i][2])
                #angle = math.radians(augmented_gknet_label[i][5])
                lx,ly = x - w/2., y
                rx,ry = x + w/2., y 
                lx,ly = dataset.rotated_about(lx,ly,x,y,angle)
                rx,ry = dataset.rotated_about(rx,ry,x,y,angle)
                
                w = gknet_label[i][4] * 224
                xt,yt = gknet_label[i][0] , gknet_label[i][1] * 224
                angle = math.atan2(gknet_label[i][3],gknet_label[i][2])
                #angle = math.radians(gknet_label[i][5])
                lxt,lyt = xt - w/2., yt
                rxt,ryt = xt + w/2., yt 
                lxt,lyt = dataset.rotated_about(lxt,lyt,xt,yt,angle)
                rxt,ryt = dataset.rotated_about(rxt,ryt,xt,yt,angle)
            
                save_name = 'vis_out/' + str(idx_plot) 
                fig, axs = plt.subplot_mosaic([['A', 'B']],
                                                    figsize=(10,5))
                
                
                axs['A'].set_title('Normal image')
                axs['B'].set_title('Augmented image')
                axs['A'].imshow(denorm_torch_to_pil(img[i]))
                axs['A'].scatter([lxt],[lyt],color='red',s=3)
                axs['A'].scatter([rxt],[ryt],color='green',s=3)
                axs['A'].scatter([xt],[yt],color='cyan',s=3)
                axs['A'].scatter([points_grasp[i][0,0]],[points_grasp[i][0,1]],color='blue',s=3)
                axs['A'].scatter([points_grasp[i][1,0]],[points_grasp[i][1,1]],color='blue',s=3)
                
                axs['B'].imshow(denorm_torch_to_pil(augmented_img[i]))
                axs['B'].scatter([lx],[ly],color='red',s=3)
                axs['B'].scatter([rx],[ry],color='green',s=3)
                axs['B'].scatter([x],[y],color='cyan',s=3)
                plt.tight_layout()
                plt.savefig(save_name + '.png', dpi=150)
                plt.close('all')
                idx_plot = idx_plot + 1
        
        gknet_label = gknet_label.to(device)
        augmented_gknet_label = augmented_gknet_label.to(device)
        img = img.to(device)
        augmented_img = augmented_img.to(device)
        
        center,theta_cos,theta_sin,w = model(img,augmented_img,gknet_label)
        
        centergt,theta_cosgt,theta_singt,wgt = augmented_gknet_label[:,:2], augmented_gknet_label[:,2],\
                                                augmented_gknet_label[:,3], augmented_gknet_label[:,4]
        
        optim.zero_grad()

        center_loss = loss(center,centergt)
        cos_loss = loss(theta_cos,theta_cosgt)
        sin_loss = loss(theta_sin,theta_singt)
        w_loss = loss(w,wgt)
        total_loss = 10 * center_loss + 10 * cos_loss + 10 * sin_loss + 10 * w_loss
        total_loss = total_loss / batch_size
        print("--------- Epoch {} ---------".format(epoch))
        print("Angle loss", (cos_loss + sin_loss)/float(batch_size) ) 
        print("Center loss", (center_loss/batch_size)) 
        print("Width Loss", w_loss/batch_size)
        print("Loss",total_loss)
        #import pdb; pdb.set_trace()
        print("Center GT ", centergt[0,0],centergt[0,1],wgt[0])
        print("Center pred", center[0,0],center[0,1],w[0])
        print("Angle GT", theta_cosgt[0],theta_singt[0])
        print("Angle pred", theta_cos[0],theta_sin[0])
        
        ##add wandb logging here
        total_loss.backward()
        optim.step()
        
        
        '''
        relevant for inference later
        best_idx, query_feats, ref_feats = model.select_best_views(all_target_images[0],ref_image)
        ref_raw_labels = ref_raw_labels[:,:,:4] #remove the last column (h) 
        target_grasp = target_raw_labels[best_idx][:,:,:4]
        #import pdb; pdb.set_trace()
        x,y,theta,w = model(query_feats,ref_feats)
        #import pdb;pdb.set_trace()
        
        save_name = 'vis_out/' + str(idx_plot) 
        fig, axs = plt.subplot_mosaic([['A', 'B']],
                                            figsize=(10,5))
        axs['A'].set_title('Reference image')
        axs['B'].set_title('Query images')
        axs['A'].imshow(denorm_torch_to_pil(ref_image[0]))
        tgt_pils = [denorm_torch_to_pil(
                    all_target_images[0][j]) for j in range(all_target_images[0].shape[0])]
        tgt_pils = tile_ims_horizontal_highlight_best(tgt_pils, highlight_idx=best_idx)
        axs['B'].imshow(tgt_pils)
        plt.tight_layout()
        plt.savefig(save_name + '.png', dpi=150)
        plt.close('all')
        idx_plot = idx_plot + 1
        '''
        