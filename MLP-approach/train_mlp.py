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
import shutil
import einops
from einops import repeat
from dataset import TestDataset
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

categories = ["backpack", "bicycle", "book", "car", "chair", "hairdryer", "handbag",
              "hydrant", "keyboard", "laptop", "motorcycle", "mouse", "remote", 
              "teddybear", "toaster", "toilet", "toybus", "toyplane", "toytrain", "toytruck"]
idx_plot = 0

image_transform = get_transform()
CROP = False 
dataset = TestDataset(image_transform=image_transform,num_targets=args.n_target,vis=True,crop=CROP)
dataset.img_cnt = 0 

vis_dir = 'vis_out/'

os.makedirs(vis_dir, exist_ok=True)



for category in ["Jacquard"] :
    
    dataloader = DataLoader(dataset, batch_size = args.batch_size,num_workers=0,shuffle=False)
    categories = dataset.classes
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        if CROP : 
            ref_image, all_target_images, mask_ref, mask_targets, grasps_ref, grasps_target,ref_path, target_path,\
                target_raw_labels, ref_raw_labels, ref_dims, target_dims, ref_gripper_pts, target_gripper_pts = batch
        else : 
            ref_image, all_target_images, mask_ref, mask_targets, grasps_ref, grasps_target,ref_path, target_path,\
                target_raw_labels, ref_raw_labels,  ref_gripper_pts, target_gripper_pts = batch
        
        #extract x,y centers of the grasping boxes
        centers_ref  = ref_raw_labels[:][:,:,0:2]
        centers_targets = [i[:,:,0:2] for i in target_raw_labels]

        batch_size = ref_image.size(0)
        #ref_image = Image.open('/home/cedric/ZeroShot-Grasping/zero-shot-pose/zsp/0/ref_crop_RGB26.png')
        #target_image = Image.open('/home/cedric/ZeroShot-Grasping/zero-shot-pose/zsp/0/target_crop_RGB_26.png')
        #all_images = torch.cat([image_transform(ref_image).unsqueeze(0).unsqueeze(0), image_transform(target_image).unsqueeze(0).unsqueeze(0)], dim=1).to(device)
        #img_clone = all_images.clone()
        #all_images = torch.cat([ref_image.unsqueeze(1), all_target_images], dim=1).to(device) # B x (N_TGT + 1) x 3 x S x S
        best_idx, query_feats, ref_feats = model.select_best_views(all_target_images[0],ref_image)
        ref_raw_labels = ref_raw_labels[:,:,:4] #remove the last column (h) 
        target_grasp = target_raw_labels[best_idx][:,:,:4]
        import pdb; pdb.set_trace()
        x,y,theta,w = model(query_feats,ref_feats)
        import pdb;pdb.set_trace()
        
        
        ##plot 
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
        