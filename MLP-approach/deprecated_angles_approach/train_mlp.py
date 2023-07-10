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
#import einops
#from einops import repeat
from dataset_reference import TestDataset
from dataset_augment import AugmentDataset
from model import GraspTransformer
from torch.utils.tensorboard import SummaryWriter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


writer = SummaryWriter()

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
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu:0')
# ---------------
# SET UP DESCRIPTOR CLASS
# ---------------
image_norm_mean = (0.485, 0.456, 0.406)
image_norm_std = (0.229, 0.224, 0.225)
image_size = 840 
    
    
    

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
folder_checkpoint = 'checkpoints/'

os.makedirs(folder_checkpoint, exist_ok=True)
# ---------------
# LOAD MODEL
# ---------------
pretrain_path = pretrain_paths[args.patch_size]



categories = ["backpack", "bicycle", "book", "car", "chair", "hairdryer", "handbag",
              "hydrant", "keyboard", "laptop", "motorcycle", "mouse", "remote", 
              "teddybear", "toaster", "toilet", "toybus", "toyplane", "toytrain", "toytruck"]
idx_plot = 0

image_transform = get_transform()
CROP = False 
TRAIN_VIS = True
ANGLE_MODE = True
MODE = 'similarity' #similiarity on or off -> off -> normal MLP is used
MODE = 'naive' #similiarity on or off -> off -> normal MLP is used
MODE = 'resnet' #similiarity on or off -> off -> normal MLP is used
MODE = 'attention'
'''
angle_mode = True : uses the angle representation (x,y,theta_cos,theta_sin,w)
angle_mode = False : uses the grasp point representation (xl,yl,xr,yr)

the model is automatically adjusted in its layer sizes when the flag is set here
'''

vis_every = 10 #vis one train sample every 30 interations

model = GraspTransformer(angle_mode=ANGLE_MODE,img_size=image_size)
model = model.to(device)
model.train()

print("load data")
dataset = TestDataset(image_transform=image_transform,num_targets=args.n_target,vis=True,crop=CROP)
dataset = AugmentDataset(image_transform=image_transform,num_targets=args.n_target,vis=True,
                         crop=CROP,overfit=False,img_size=image_size)
dataset.img_cnt = 0 

vis_dir = 'vis_out/'  #to look at the labels
train_vis_dir =  'train_vis/' #for labels and prediction visualisation 
train_plots = 0 #counter to vis predictions 

if os.path.exists(train_vis_dir): 
    shutil.rmtree(train_vis_dir)

os.makedirs(vis_dir, exist_ok=True)
os.makedirs(train_vis_dir, exist_ok=True)

optim = torch.optim.Adam(model.parameters(), lr=1e-3)
loss = torch.nn.MSELoss()

epochs = 200
n_iter = 0 
for epoch in range(epochs) :
    
    dataloader = DataLoader(dataset, batch_size = args.batch_size,num_workers=0,shuffle=False)
    categories = dataset.classes
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        data = batch
        
        img = data['img']
        augmented_img = data['img_augmented']
        gknet_label = data['img_grasp']
        augmented_gknet_label = data['img_augmented_grasp']
        augmented_gknet_label2 = data['img_augmented_grasp2']
        batch_size = img.shape[0]
        points_grasp = data['points_grasp']
        points_grasp_augmented = data['points_grasp_augmented']
        points_grasp_augmented2 = data['points_grasp_augmented2']
        ##for the visualisation of the grasp 
        VIS = False
        if VIS : 
            for i in range(batch_size):
                w = augmented_gknet_label[i][4] * image_size
                x,y = augmented_gknet_label[i][0] *  image_size, augmented_gknet_label[i][1] * image_size
                angle = math.atan2(augmented_gknet_label[i][3],augmented_gknet_label[i][2])
                #angle = math.radians(augmented_gknet_label[i][5])
                lx,ly = x - w/2., y
                rx,ry = x + w/2., y 
                lx,ly = dataset.rotated_about(lx,ly,x,y,angle)
                rx,ry = dataset.rotated_about(rx,ry,x,y,angle)
                
                w = gknet_label[i][4] * image_size
                xt,yt = gknet_label[i][0] *  image_size, gknet_label[i][1] * image_size
                angle = math.atan2(gknet_label[i][3],gknet_label[i][2])
        
                #angle = math.radians(gknet_label[i][5])
                lxt,lyt = xt - w/2., yt
                rxt,ryt = xt + w/2., yt 
                lxt,lyt = dataset.rotated_about(lxt,lyt,xt,yt,angle)
                rxt,ryt = dataset.rotated_about(rxt,ryt,xt,yt,angle)
            
                save_name = vis_dir + str(idx_plot) 
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
                axs['B'].scatter([points_grasp_augmented[i][0,0]],[points_grasp_augmented[i][0,1]],color='blue',s=3)
                axs['B'].scatter([points_grasp_augmented[i][1,0]],[points_grasp_augmented[i][1,1]],color='blue',s=3)
                plt.tight_layout()
                plt.savefig(save_name + '.png', dpi=150)
                plt.close('all')
                idx_plot = idx_plot + 1
        
        gknet_label = gknet_label.to(device)
        augmented_gknet_label = augmented_gknet_label.to(device)
        augmented_gknet_label2 = augmented_gknet_label2.to(device)
        img = img.to(device)
        augmented_img = augmented_img.to(device)
        points_grasp = points_grasp.to(device)
        points_grasp_augmented = points_grasp_augmented.to(device)
        points_grasp_augmented2 = points_grasp_augmented2.to(device)
        
        if ANGLE_MODE == True : 
            
            if MODE == 'similarity' :
                center_pred,theta_cos_pred,theta_sin_pred,w_pred,img_feats, augemented_feats = model.forward_similarity(img,augmented_img,gknet_label)
            elif MODE == 'naive' : 
                center_pred,theta_cos_pred,theta_sin_pred,w_pred,img_feats, augemented_feats = model.forward_naive(img,augmented_img,gknet_label)
            elif MODE == 'resnet': 
                center_pred,theta_cos_pred,theta_sin_pred,w_pred,img_feats, augemented_feats = model.forward_pca(img,augmented_img,gknet_label)
            elif MODE == 'attention': 
                center_pred,theta_cos_pred,theta_sin_pred,w_pred,img_feats, augemented_feats = model.forward_attention(img,augmented_img,gknet_label)
            
            centergt,theta_cosgt,theta_singt,wgt = augmented_gknet_label[:,:2], augmented_gknet_label[:,2],\
                                                    augmented_gknet_label[:,3], augmented_gknet_label[:,4]

            theta_singt2, theta_cosgt2 = augmented_gknet_label2[:,2], augmented_gknet_label2[:,3]
            
            
        else : 
            if MODE == 'similarity' :
                point_left_pred, point_right_pred, img_feats, augemented_feats  = model.forward_similarity(img,augmented_img,points_grasp.reshape(points_grasp.shape[0],4))
            elif MODE == 'naive' : 
                point_left_pred, point_right_pred, img_feats, augemented_feats  = model.forward_naive(img,augmented_img,points_grasp.reshape(points_grasp.shape[0],4))
            elif MODE == 'resnet': 
                point_left_pred, point_right_pred, img_feats, augemented_feats = model.forward_pca(img,augmented_img,gknet_label)
            point_leftgt, point_rightgt = points_grasp_augmented[:,0].to(torch.float32) /  image_size, points_grasp_augmented[:,1].to(torch.float32)/  image_size
            point_leftgt2, point_rightgt2 = points_grasp_augmented2[:,0].to(torch.float32) /  image_size, points_grasp_augmented2[:,1].to(torch.float32)/  image_size
            
        optim.zero_grad()

        if ANGLE_MODE == True : 
            center_loss = loss(center_pred,centergt)
            
            cos_loss = loss(theta_cos_pred,theta_cosgt)
            cos_loss2 = loss(theta_cos_pred,theta_cosgt2)
            sin_loss = loss(theta_sin_pred,theta_singt)
            sin_loss2 = loss(theta_sin_pred,theta_singt2)
            
            angle_loss1 = cos_loss + sin_loss
            angle_loss2 = cos_loss2 + sin_loss2
            
            angle_loss = min(angle_loss1,angle_loss2)
            w_loss = loss(w_pred,wgt)
            #total_loss = center_loss
            total_loss = 10 * center_loss + angle_loss +  w_loss * 10
            total_loss = total_loss / batch_size
            print("--------- Epoch {} ---------".format(epoch))
            print("Angle loss", (cos_loss + sin_loss)/float(batch_size) ) 
            print("Center loss", (center_loss/batch_size)) 
            print("Width Loss", w_loss/batch_size)
            print("Loss",total_loss)
            
            writer.add_scalar('Loss/train',total_loss , n_iter)
            writer.add_scalar('AngleLoss/train', (cos_loss + sin_loss)/batch_size, n_iter)
            writer.add_scalar('CenterLoss/train',center_loss/batch_size , n_iter)
            writer.add_scalar('WidthLoss/train',w_loss/batch_size , n_iter)
        else : 
            ptleft_loss = loss(point_left_pred,point_leftgt) / float(batch_size)
            ptright_loss = loss(point_right_pred,point_rightgt) / float(batch_size)
            point_loss1 = ptleft_loss + ptright_loss
            
            ptleft_loss = loss(point_left_pred,point_leftgt2) / float(batch_size)
            ptright_loss = loss(point_right_pred,point_rightgt2) / float(batch_size)
            point_loss2 = ptleft_loss + ptright_loss
            
            total_loss = min(point_loss1,point_loss2)
            print("--------- Epoch {} ---------".format(epoch))
            print("Left Point loss", ptleft_loss ) 
            print("Right Point loss",ptright_loss) 
            print("Loss",total_loss)
            
            writer.add_scalar('Loss/train',total_loss , n_iter)
            writer.add_scalar('LeftPointLoss/train', ptleft_loss, n_iter)
            writer.add_scalar('RightPointLoss/train', ptright_loss, n_iter)

        n_iter = n_iter + 1
        total_loss.backward()
        optim.step()
        
        if TRAIN_VIS and n_iter % vis_every == 0 :
            patch_h, patch_w = int(image_size/14.), int(image_size/14.)
            i = 0 
            w = augmented_gknet_label[i][4] * image_size
            x,y = augmented_gknet_label[i][0] *  image_size, augmented_gknet_label[i][1] * image_size
            angle = math.atan2(augmented_gknet_label[i][3],augmented_gknet_label[i][2])
            angle2 = math.atan2(augmented_gknet_label2[i][3],augmented_gknet_label2[i][2])
            #angle = math.radians(augmented_gknet_label[i][5])
            lx1,ly1 = x - w/2., y
            rx1,ry1 = x + w/2., y 
            lx,ly = dataset.rotated_about(lx1,ly1,x,y,angle)
            rx,ry = dataset.rotated_about(rx1,ry1,x,y,angle)
            
            lx2,ly2 = dataset.rotated_about(lx1,ly1,x,y,angle2)
            rx2,ry2 = dataset.rotated_about(rx1,ry1,x,y,angle2)
            
            
            
            if ANGLE_MODE == True :
                wp = w_pred[i] * image_size
                xt,yt = center_pred[i][0] *  image_size ,center_pred[i][1] * image_size
                angle = math.atan2(theta_sin_pred[i],theta_cos_pred[i])
                #angle = math.radians(gknet_label[i][5])
                lxt,lyt = xt - wp/2., yt
                rxt,ryt = xt + wp/2., yt 
                lxt,lyt = dataset.rotated_about(lxt,lyt,xt,yt,angle)
                rxt,ryt = dataset.rotated_about(rxt,ryt,xt,yt,angle)
            else : 
                lx,ly = point_leftgt[i][0] *  image_size ,point_leftgt[i][1] * image_size
                rx,ry = point_rightgt[i][0] *  image_size ,point_rightgt[i][1] * image_size
                lx2,ly2 = point_leftgt2[i][0] *  image_size ,point_leftgt2[i][1] * image_size
                rx2,ry2 = point_rightgt2[i][0] *  image_size ,point_rightgt2[i][1] * image_size
                rx = rx.cpu().detach()
                ry = ry.cpu().detach()
                lx = lx.cpu().detach()
                ly = ly.cpu().detach()
                
                rx2 = rx2.cpu().detach()
                ry2 = ry2.cpu().detach()
                lx2 = lx2.cpu().detach()
                ly2 = ly2.cpu().detach()
                
                
                lxt,lyt = point_left_pred[i][0] *  image_size, point_left_pred[i][1] * image_size
                rxt, ryt = point_right_pred[i][0] *  image_size ,point_right_pred[i][1] * image_size
                lxt, lyt = lxt.cpu().detach(), lyt.cpu().detach()
                rxt, ryt = rxt.cpu().detach(), ryt.cpu().detach()
        
            save_name = train_vis_dir + str(train_plots) 
            
                    
            
            
            fig, axs = plt.subplot_mosaic([['A', 'B',"C",],
                                            ["D","E","F"]],
                                                figsize=(10,5))
            axs['A'].set_title('Augmented image Preds')
            axs['B'].set_title('Augmented image GT')
            axs['C'].set_title('Image features')
            axs['C'].set_title('Normal Image features')
            axs['D'].set_title('Augmented Image features')
            axs['E'].set_title('Image features foreground')
            axs['F'].set_title('Augmented features foreground')
            axs['A'].imshow(denorm_torch_to_pil(augmented_img[i].cpu()))
            axs['A'].scatter([lxt],[lyt],color='red',s=3)
            axs['A'].scatter([rxt],[ryt],color='green',s=3)
            if ANGLE_MODE == True :
                axs['A'].scatter([xt.cpu().detach()],[yt.cpu().detach()],color='cyan',s=3)
            #axs['A'].scatter([points_grasp[i][0,0]],[points_grasp[i][0,1]],color='blue',s=3)
            #axs['A'].scatter([points_grasp[i][1,0]],[points_grasp[i][1,1]],color='blue',s=3)
            
            axs['B'].imshow(denorm_torch_to_pil(augmented_img[i].cpu()))
            axs['B'].scatter([lx],[ly],color='red',s=3)
            axs['B'].scatter([rx],[ry],color='green',s=3)
            axs['B'].scatter([lx2],[ly2],color='red',s=3)
            axs['B'].scatter([rx2],[ry2],color='green',s=3)
            if ANGLE_MODE == True :
                axs['B'].scatter([x.cpu().detach()],[y.cpu().detach()],color='cyan',s=3)
                
            #attentions = model.dinov2d_backbone.get_last_selfattention(img) 
            
            if MODE == "resnet":
                img_feats = img_feats[i].cpu()
                augmented_img_feats = augemented_feats[i].cpu()
                total_features = torch.stack([img_feats, augmented_img_feats]).reshape(-1,384).numpy()
                pca = PCA(n_components=3,random_state=42)
                pca.fit(total_features)
                pca_features = pca.transform(total_features)
                pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()) / \
                        (pca_features[:, 0].max() - pca_features[:, 0].min())

                pca_features_img = pca_features.reshape(2, patch_h, patch_w, 3)
                
                img1 = pca_features[i * patch_h * patch_w: (i+1) * patch_h * patch_w,0].reshape(patch_h, patch_w)
                img2 = pca_features[1 * patch_h * patch_w: (1+1) * patch_h * patch_w,0].reshape(patch_h, patch_w)
                axs['C'].imshow(img1) 
                axs['D'].imshow(img2) 
                pca_features_bg = pca_features[:, 0] <= 0.35 # from first histogram
                pca_features_fg = ~pca_features_bg
                bg_num = sum(pca_features_bg)
                fg_num = sum(pca_features_fg)
                #if bg_num > fg_num:
                #    pca_features_bg, pca_features_fg = pca_features_fg, pca_features_bg 
                #import pdb; pdb.set_trace()
                pca.fit(total_features[pca_features_fg]) 
                pca_features_left = pca.transform(total_features[pca_features_fg])

                for i in range(3):
                    # min_max scaling
                    pca_features_left[:, i] = (pca_features_left[:, i] - pca_features_left[:, i].min()) / (pca_features_left[:, i].max() - pca_features_left[:, i].min())

                pca_features_rgb = pca_features.copy()
                # for black background
                pca_features_rgb[pca_features_bg] = 0
                # new scaled foreground features
                pca_features_rgb[pca_features_fg] = pca_features_left
                pca_features_rgb = pca_features_rgb.reshape(2, patch_h, patch_w, 3)
                
                axs['E'].imshow(pca_features_rgb[0]) 
                axs['F'].imshow(pca_features_rgb[1]) 
            
            elif MODE == 'attention':
                axs['C'].imshow(img_feats[0,0].cpu().numpy()) 
                axs['D'].imshow(augemented_feats[0,0].cpu().numpy()) 
                
                axs['E'].imshow(img_feats[0,1].cpu().numpy()) 
                axs['F'].imshow(augemented_feats[0,1].cpu().numpy()) 
            '''
            pca = PCA(n_components=3)
            pca.fit(img_feats)
            pca_features = pca.transform(img_feats)
            pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()) / \
                    (pca_features[:, 0].max() - pca_features[:, 0].min())
            pca_features_bg = pca_features[:, 0] > 0.35 # from first histogram
            pca_features_fg = ~pca_features_bg
            
            pca.fit(img_feats[pca_features_fg]) 
            pca_features_left = pca.transform(img_feats[pca_features_fg])
            for id in range(3):
                # min_max scaling
                pca_features_left[:, id] = (pca_features_left[:, id] - pca_features_left[:, id].min()) / (pca_features_left[:, id].max() - pca_features_left[:, id].min())

            pca_features_rgb = pca_features.copy()
            # for black background
            pca_features_rgb[pca_features_bg] = 0
            # new scaled foreground features
            pca_features_rgb[pca_features_fg] = pca_features_left
            pca_features_rgb = pca_features_rgb.reshape(patch_h, patch_w, 3)
            '''
            
            #axs['C'].imshow(pca_features[i*patch_h*patch_w : (i+1)*patch_h*patch_w, 0].reshape(patch_h, patch_w))    
            #axs['E'].imshow(pca_features_rgb)
            #augmented_img_feats = augemented_feats[i].cpu()
            '''
            pca = PCA(n_components=3)
            pca.fit(augmented_img_feats)
            pca_features = pca.transform(augmented_img_feats)
            pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()) / \
                    (pca_features[:, 0].max() - pca_features[:, 0].min())
            
            pca_features_bg = pca_features[:, 0] > 0.35 # from first histogram
            pca_features_fg = ~pca_features_bg
            
            pca.fit(img_feats[pca_features_fg]) 
            pca_features_left = pca.transform(img_feats[pca_features_fg])
            for id in range(3):
                # min_max scaling
                pca_features_left[:, id] = (pca_features_left[:, id] - pca_features_left[:, id].min()) / (pca_features_left[:, id].max() - pca_features_left[:, id].min())

            pca_features_rgb = pca_features.copy()
            # for black background
            pca_features_rgb[pca_features_bg] = 0
            # new scaled foreground features
            pca_features_rgb[pca_features_fg] = pca_features_left
            pca_features_rgb = pca_features_rgb.reshape(patch_h, patch_w, 3)
            '''
            #axs['F'].imshow(pca_features_rgb)
            #axs['D'].imshow(pca_features[i*patch_h*patch_w : (i+1)*patch_h*patch_w, 0].reshape(patch_h, patch_w))   
            
            plt.tight_layout()
            plt.savefig(save_name + '.png', dpi=150)
            plt.close('all')
            train_plots = train_plots + 1
            
        #import pdb; pdb.set_trace()
        #print("Center GT ", centergt[0,0],centergt[0,1],wgt[0])
        #print("Center pred", center[0,0],center[0,1],w[0])
        #print("Angle GT", theta_cosgt[0],theta_singt[0])
        #print("Angle pred", theta_cos[0],theta_sin[0])
        
        ##add wandb logging here
        
    
    torch.save(model.state_dict(),folder_checkpoint + 'model_similarity_epoch{}.pt'.format(epoch))
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
    
        