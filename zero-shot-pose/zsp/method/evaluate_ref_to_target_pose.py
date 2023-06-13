import os
import argparse

import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from pytorch3d.ops import utils as oputil
from pytorch3d.renderer.cameras import get_world_to_view_transform
import torchvision 
from PIL import Image
from matplotlib.patches import Polygon
import cv2

# Descriptor extractor 
from zsp.method.zero_shot_pose import DescriptorExtractor, ZeroShotPoseMethod
# Datasets and models
from zsp.datasets.co3d_pose_dataset import Co3DPoseDataset, co3d_pose_dataset_collate, TestDataset
# Utils
from zsp.method.zero_shot_pose_utils import (
    scale_points_to_orig,
    get_structured_pcd, 
    trans21_error, 
)
from zsp.utils.paths import LABELS_DIR, DETERM_EVAL_SETS, LOG_DIR

from zsp.utils.project_utils import AverageMeter
from zsp.utils.depthproc import transform_cameraframe_to_screen
from zsp.utils.visuals import (
    plot_pcd,
    tile_ims_horizontal_highlight_best,
    draw_correspondences_lines
)

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
desc = DescriptorExtractor(
    patch_size=args.patch_size,
    feat_layer=9,
    high_res=False,
    binning=args.binning,
    image_size=224,
    n_target=args.n_target,
    saliency_map_thresh=0.1,
    num_correspondences=args.num_correspondences,
    kmeans=args.kmeans,
    best_frame_mode=args.best_frame_mode
)
# ---------------
# SET UP ZERO-SHOT POSE CLASS
# ---------------
pose = ZeroShotPoseMethod(
    batched_correspond=True,
    num_plot_examples_per_batch=1,
    saliency_map_thresh=0.1,
    ransac_thresh=args.ransac_thresh,
    n_target=args.n_target,
    num_correspondences=args.num_correspondences,
    take_best_view=args.take_best_view,
)

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
label_dir = LABELS_DIR
os.makedirs(log_dir, exist_ok=True)

determ_eval_root = os.path.join(DETERM_EVAL_SETS, f'200samples_{args.n_target}tgt')
pretrain_paths = {
    16: os.path.join(args.hub_dir, 'dino_vitbase16_pretrain.pth'),
    8: os.path.join(args.hub_dir, 'dino_deitsmall8_pretrain.pth'),
}


kmeans_str = 'Kmeans' if args.kmeans else 'NoKmeans'
note = args.note
log_dir = os.path.join(log_dir,
                       f"{args.num_correspondences}corr_{kmeans_str}_{args.binning}bin_{args.ransac_thresh}ransac_{args.n_target}tgt_{note}")

# ---------------
# LOAD MODEL
# ---------------
pretrain_path = pretrain_paths[args.patch_size]
desc.load_model(pretrain_path, device)

categories = ["backpack", "bicycle", "book", "car", "chair", "hairdryer", "handbag",
              "hydrant", "keyboard", "laptop", "motorcycle", "mouse", "remote", 
              "teddybear", "toaster", "toilet", "toybus", "toyplane", "toytrain", "toytruck"]
idx_plot = 0

image_transform = desc.get_transform()
CROP = False 
dataset = TestDataset(image_transform=image_transform,num_targets=args.n_target,vis=True,crop=CROP)
dataset.img_cnt = 0 

vis_dir = 'vis_out/'

os.makedirs(vis_dir, exist_ok=True)

for category in ["Jacquard"] :
    cat_log_dir, fig_dir = pose.make_log_dirs(log_dir, category)
    # ---------------
    # LOAD DATASET
    # ---------------
    #dataset = Co3DPoseDataset(dataset_root=co3d_root,
    #                        categories=[category],
    #                        num_samples_per_class=args.num_samples_per_class,
    #                        target_frames_sampling_mode='uniform',
    #                        num_frames_in_target_seq=pose.n_target,
    #                        label_dir=label_dir,
    #                        determ_eval_root=determ_eval_root,
    #                        image_transform=image_transform)
    
    
    
    
    #dataloader = DataLoader(dataset, num_workers=args.num_workers,
    #                        batch_size=args.batch_size, collate_fn=co3d_pose_dataset_collate, shuffle=False)
    
    dataloader = DataLoader(dataset, batch_size = args.batch_size,num_workers=0,shuffle=False)
    categories = dataset.classes
    loggers = {}
    for c in categories:
        loggers[c] = {
            'rot_error': AverageMeter(),
            'acc_30': AverageMeter(),
        }
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
        all_images = torch.cat([ref_image.unsqueeze(1), all_target_images], dim=1).to(device) # B x (N_TGT + 1) x 3 x S x S
        # Extract features, attention maps, and cls_tokens
        '''
        all_images is of shape 2, 6, 3, 224, 224
        2 is the batch size here
        ref_image : one image of shape 3, 224, 224
        all_target_images : 5 images of shape 3, 224, 224
        '''
        features, attn, output_cls_tokens = desc.extract_features_and_attn(all_images)
        # Create descriptors from features, return descriptors and attn in appropriate shapes
        # attn shape Bx(n_tgt+1)xhxtxt, features shape Bx(n_tgt+1)x1x(t-1)xfeat_dim
        features, attn = desc.create_reshape_descriptors(features, attn, batch_size, device)
        # Split ref/target, repeat ref to match size of target, and flatten into batch dimension
        #print('features shape',features.shape)
        #print('attn shape',attn.shape)
        try :
            ref_feats, target_feats, ref_attn, target_attn = desc.split_ref_target(features, attn)
        except :
            ## sometimes an issue can occur 
            continue
        
        # ----------------
        # GET CORRESPONDENCES
        # ----------------
        (selected_points_image_2, # 10x50x2
            selected_points_image_1, # 10x50x2
            cyclical_dists,          # 10x28x28
            sim_selected_12) = desc.get_correspondences(ref_feats, target_feats, ref_attn, target_attn, device)
        
        new_arr_1 = []
        new_arr_2 = []

        #  sim_selected_12 has shape 10x50
        # ----------------
        # FIND BEST IMAGE IN TARGET SEQ
        # ----------------
        _, _, _, t, t = attn.size()
        N = int(np.sqrt(t - 1)) # N is the height or width of the feature map
        similarities, best_idxs = desc.find_closest_match(attn, output_cls_tokens, sim_selected_12, batch_size)
        #import pdb; pdb.set_trace() 
        # -----------------
        # COMPUTE POSE OFFSET
        # -----------------
        all_trans21 = []
        all_errs = []
        all_points1 = []
        all_points2 = []
        for i in range(batch_size):
            # -----------------
            # PREPARE DATA
            # -----------------
            # Get other data required to compute pose offset
            target_frame = best_idxs[i]

            # NB: from here on, '1' <--> ref and '2' <--> target
            # Get points and if necessary scale them from patch to pixel space
            points1, points2 = (
                selected_points_image_1[i*args.n_target + target_frame],
                selected_points_image_2[i*args.n_target + target_frame]
            )
            points1_rescaled, points2_rescaled = desc.scale_patch_to_pix(
                points1, points2, N
            )
            all_points1.append(points1_rescaled.clone().int().long())
            all_points2.append(points2_rescaled.clone().int().long())
        
        for i in range(len(all_points1)):
                    for j,pts in enumerate(all_points1[i]) :
                            x,y = pts
                            mask_value = mask_ref[0,0,x,y]
                            #import pdb; pdb.set_trace()
                            #print(mask_value)
                            #if mask_value.item() == 0 :
                            #    #import pdb; pdb.set_trace()        
                            #    #set correspondences that are outside of the object area to 0 
                            #    # right now : ref mask is used, could be changed to query mask later 
                            #    # (unless we have segmentation for unknown object)
                            #    all_points1[i][j] = torch.Tensor([-1,-1])
                            #    all_points2[i][j] = torch.Tensor([-1,-1])
        
        
        
        out1 = all_points1[0][all_points1[0][:,0] != -1].numpy().reshape(-1,1,2).astype(np.float32)
        out2 = all_points2[0][all_points2[0][:,0] != -1].numpy().reshape(-1,1,2).astype(np.float32)
        
        matrix, mask = cv2.findHomography(out2, out1, cv2.RANSAC, 5.0)
        # applying perspective algorithm
        dst = cv2.perspectiveTransform(out2.reshape(1,-1,2), matrix)
        
        ## the two grasping points
        #import pdb; pdb.set_trace()
        target_pts = torch.Tensor(target_gripper_pts[best_idxs[0]])
        target_pts = target_pts.numpy().reshape(1,-1,2).astype(np.float32)
        target_pts[:,:,0], target_pts[:,:,1] = target_pts[:,:,1].copy(), target_pts[:,:,0].copy()
        new_pts = cv2.perspectiveTransform(target_pts, matrix).reshape(-1,2,2)
        
        #switch x,y axes as this is the order in the correspondence points
        grasps_new = grasps_target[best_idxs[0]].numpy().copy()
        #grasps_new = grasps_ref.numpy().copy()
        #import pdb; pdb.set_trace()
        grasps_new[0][:,:,0] = grasps_target[best_idxs[0]][0][:,:,1]
        grasps_new[0][:,:,1] = grasps_target[best_idxs[0]][0][:,:,0]
        
        #import pdb; pdb.set_trace()
        grasp_new = rotate_grasping_rectangle(grasps_new,matrix)
        
        #transform grasp center to the new frame 
        centers_targets = centers_targets[best_idxs[0]].numpy().astype(np.float32).reshape(-1,1,2)
        centers_ref = centers_ref.numpy().astype(np.float32).reshape(-1,1,2)
        
        centers_new = centers_ref.copy()
        centers_new[:,:,0] = centers_ref[:,:,1]
        centers_new[:,:,1] = centers_ref[:,:,0]
        centers_new = cv2.perspectiveTransform(centers_new, matrix)
        
        
        
        
        if plot_results:

            # -----------------
            # PLOTTING
            # -----------------
            for i in range(args.num_plot_examples_per_batch):
                #save_name = f'{all_errs[i]}_err_{category}'
                save_name = f'{idx_plot}_err_{category}'
                idx_plot += 1
                save_name = os.path.join(fig_dir, save_name)

                fig, axs = plt.subplot_mosaic([['A', 'B', 'B','C'],
                                            ['D','D','E','F']],
                                            figsize=(10,5))
                for ax in axs.values():
                    ax.axis('off')
                axs['A'].set_title('Reference image')
                axs['B'].set_title('Query images')
                axs['C'].set_title('Correspondences')
                axs['D'].set_title('Reference object mask')
                axs['E'].set_title('Point Transform Comparison')
                axs['F'].set_title('Transform of Grasps')
                fig.suptitle(f'Error: n/a', fontsize=6)
                axs['A'].imshow(desc.denorm_torch_to_pil(ref_image[i]))
                
                # ax[1].plot(similarities[i].cpu().numpy())
                tgt_pils = [desc.denorm_torch_to_pil(
                    all_target_images[i][j]) for j in range(args.n_target)]
                tgt_pils = tile_ims_horizontal_highlight_best(tgt_pils, highlight_idx=best_idxs[i])
                axs['B'].imshow(tgt_pils)
                print(desc.torch_to_pil(mask_ref[0]).shape)
                axs['D'].imshow(desc.torch_to_pil(mask_ref[0]),cmap='Greys_r')
                idcs = (mask_ref[0] != 0).nonzero(as_tuple=False)[:,1:3]
                max_x,min_x,max_y,min_y = idcs[:,0].max(),idcs[:,0].min(),idcs[:,1].max(),idcs[:,1].min()
                axs['D'].scatter([min_y],[min_x],color='cyan',s=3)
                axs['D'].scatter([max_y],[max_x],color='cyan',s=3)
                axs['D'].scatter([min_y],[max_x],color='cyan',s=3)
                axs['D'].scatter([max_y],[min_x],color='cyan',s=3)
                
                for pts in all_points1[i]:
                    x1,y1 = pts
                    if x1 < 0 : 
                        continue
                    axs['D'].scatter([y1],[x1],color='blue',s=1)

                axs['E'].imshow(desc.denorm_torch_to_pil(all_target_images[i][best_idxs[i]]))
                for q in range(out2.shape[0]):
                    x1,y1 = out2[q][0]
                    xpred, ypred = dst[0][q]
                    axs['E'].scatter([y1],[x1],color='green',s=1)
                    axs['E'].scatter([ypred],[xpred],color='blue',s=1)
                    
                axs['F'].imshow(desc.denorm_torch_to_pil(all_target_images[i][best_idxs[i]]))
                '''
                for idx,grasp in enumerate(grasp_new):
                    #x1,y1 = out2[q][0]
                    #xpred, ypred = dst[0][q]
                    bl,tl,tr,br = grasp
                    gt_grasp = grasps_target[best_idxs[i]][0][idx].numpy()
                    bl_gt, tl_gt,tr_gt, br_gt = gt_grasp
                    #import pdb; pdb.set_trace()
                    axs['F'].scatter([bl_gt[0]],[bl_gt[1]],color='green')
                    axs['F'].scatter([tl_gt[0]],[tl_gt[1]],color='green')
                    axs['F'].scatter([br_gt[0]],[br_gt[1]],color='green')
                    axs['F'].scatter([tr_gt[0]],[tr_gt[1]],color='green')
                    
                    axs['F'].scatter([bl[0]],[bl[1]],color='blue')
                    axs['F'].scatter([tl[0]],[tl[1]],color='blue')
                    axs['F'].scatter([br[0]],[br[1]],color='blue')
                    axs['F'].scatter([tr[0]],[tr[1]],color='blue')
                '''
                if CROP == True : 
                    dataset.visualize_imgs(target_path[best_idxs[i]][0],target_raw_labels[best_idxs[i]],
                                        grasp_new,ref_raw_labels[0],ref_path,dst,centers_new,
                                        new_pts,store_dir=vis_dir,dims=target_dims[best_idxs[i]],dims_ref=ref_dims)
                else : 
                    dataset.visualize_imgs(target_path[best_idxs[i]][0],target_raw_labels[best_idxs[i]],
                                        grasp_new,ref_raw_labels[0],ref_path,dst,centers_new,
                                        new_pts,store_dir=vis_dir)
                        
                #points = np.array([bl,tl,tr,br])
                #points_gt = np.array([bl_gt,tl_gt,tr_gt,br_gt])
                #import pdb; pdb.set_trace()
                
                #p = Polygon(points_gt, facecolor = 'k')
                #axs['F'].add_patch(p)
                #break 
                #import pdb; pdb.set_trace()
                #axs['E'].scatter([y1],[x1],color='green')
                #axs['E'].scatter([ypred],[xpred],color='blue')
                    
                draw_correspondences_lines(all_points1[i], all_points2[i],
                                        desc.denorm_torch_to_pil(ref_image[i]),
                                        desc.denorm_torch_to_pil(all_target_images[i][best_idxs[i]]),
                                        axs['C'])
                '''
                pcd1 = ref_meta_data['pcd'][i]
                pcd2 = target_meta_data['pcd'][i]
                trans21 = all_trans21[i]
                cam1 = ref_meta_data['cameras'][i]
                trans = cam1.get_world_to_view_transform().compose(trans21)
                X1, numpoints1 = oputil.convert_pointclouds_to_tensor(pcd1)
                trans_X1 = trans.transform_points(X1)
                # Project the cam2points to NDC+then into screen
                P_im = transform_cameraframe_to_screen(cam1, trans_X1, image_size=(500,500))
                P_im = P_im.squeeze()
                plot_pcd(P_im, pcd1, axs['D'])
                '''
                plt.tight_layout()
                plt.savefig(save_name + '.png', dpi=150)
                plt.close('all')

    # -----------------
    # PRINT LOGS
    # -----------------
    '''
    average_err = []
    average_acc_30 = []
    print(f'Per Class Results:')
    for cls, meters in loggers.items():
        rot_err = meters['rot_error'].avg.item()
        acc_30 = meters['acc_30'].avg
        average_err.append(rot_err)
        average_acc_30.append(acc_30)
        print(f'{cls} | Rotation Error: {rot_err:.2f} | Acc 30: {acc_30:.2f}')
    print(f'Averaged Results | Rotation Error: {np.mean(average_err):.2f} | Acc 30: {np.mean(average_acc_30):.2f}')
    '''