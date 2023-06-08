# ML
from torch.utils.data import Dataset
import torch
import numpy as np
import random
from torchvision import transforms
import torchvision.transforms.functional as TF
import math
import cv2
import shutil
# I/O
import os
import json
from PIL import Image, ImageDraw

# Typing
import math 
from typing import List

co3d_root = os.path.expanduser("~") + '/ZeroShot-Grasping/zero-shot-pose/co3d/'
label_dir = os.path.expanduser("~") + '/ZeroShot-Grasping/zero-shot-pose/data/class_labels/'
jacquard_root  = os.path.expanduser("~") + '/ZeroShot-Grasping/zero-shot-pose/Jacquard/Samples/'

def augment_image(image,angle=10):
        ##rotate the image
        image_augmented = TF.rotate(image, angle)
        
        return image_augmented


class AugmentDataset(Dataset):
    '''
    very simple dataloader to test some stuff on Jacquard samples dataset
    '''
    
    def __init__(self,dataset_root=jacquard_root,
                 image_transform=None,
                 num_targets=1,vis=False,crop=True):
        self.dataset_root = dataset_root
        self.crop = crop
        self.border_size = 100
        self.image_transform = image_transform
        self.image_norm_mean = (0.485, 0.456, 0.406)
        self.image_norm_std = (0.229, 0.224, 0.225)
        self.img_vis_dir = 'labels_vis/'
        self.classes = os.listdir(dataset_root)
        #if self.img_vis_dir[:-1] in self.classes : 
        #    self.classes.remove(self.img_vis_dir[:-1])
        self.image_dims = []
        self.items = []
        self.transform_tensor = transforms.Compose([transforms.Resize((224,224)),
                                                    transforms.ToTensor()])
        
        
        for cat in self.classes :
            if os.path.isdir(self.dataset_root + cat) == False:
                continue
            cur_dict = {}
            out_grasps = []
            fs = os.listdir(self.dataset_root + cat)
            
            imgs = [self.dataset_root + cat + "/" + i for i in fs if i.endswith('.jpg') or i.endswith('.png')]
            grasp_txts = [self.dataset_root + cat + "/" + i for i in fs if i.endswith('.txt')]
            masks = [i.split('/')[-1] for i in imgs if 'mask' in i]
            grasp_txts = [i.split('/')[-1] for i in grasp_txts]
            imgs = [i.split('/')[-1] for i in imgs if 'RGB' in i]
            
            imgs = sorted(imgs, key=lambda x: int(x.split('_')[0]))
            masks = sorted(masks, key=lambda x: int(x.split('_')[0]))
            grasp_txts = sorted(grasp_txts, key=lambda x: int(x.split('_')[0]))
            masks = [self.dataset_root + cat + "/" + i for i in masks]
            imgs = [self.dataset_root + cat + "/" + i for i in imgs]
            img_paths = imgs.copy()
            grasp_txts = [self.dataset_root + cat + "/" + i for i in grasp_txts]
            grasps = []
            self.img_cnt = 0
            store_dir = self.dataset_root + cat + '/' + self.img_vis_dir
            if os.path.exists(store_dir) == True: 
                shutil.rmtree(store_dir)
            os.makedirs(store_dir)
            grasping_labels = []
            raw_grasp_labels = []
            gripper_labels = [] 
            
            
            for idx,txt in enumerate(grasp_txts):
                grasp = []
                mids = []
                with open(txt,'r') as f:
                    lines = f.readlines()
                    for l in lines :
                        split = l.split(';')
                        x,y,angle,w,h = split 
                        if [x,y] in mids :
                            continue
                        mids.append([x,y])
                        h = h.split('\n')[0]
                        x,y,angle,w,h = float(x),float(y),float(angle),float(w),float(h)
                        if self.crop == False : 
                            grasp.append([x * 224/1024.,y * 224/1024.,angle,w * 224/1024,h * 224/1024])
                        else : 
                            grasp.append([x,y,angle,w,h])
                    
                    raw_grasp_labels.append(grasp)
                    if self.crop == False:
                        grasps, raw_labels, gripper_points = self.create_grasp_rectangle(grasp)
                        grasping_labels.append(grasps)
                        gripper_labels.append(gripper_points)
                        out_grasps.append(grasp)
                
                    
                        
            image_dict = {} #store the image dimensions for cropping
            image_dict['dims'] = []
            crop_dims = []
            ## crop the images to only the are where the object is present plus some boarder area
            if self.crop :
                
                ## for target masks
                cur_dict['crop_target'] = []
                for i in range(0,len(masks)):
                    mask_ref = self.transform_tensor(Image.open(masks[i]))
                    idcs = (mask_ref != 0).nonzero(as_tuple=False)[:,1:3]
                    max_y,min_y,max_x,min_x = idcs[:,0].max()* 1024/224.,idcs[:,0].min()* 1024/224.,idcs[:,1].max()* 1024/224.,idcs[:,1].min()* 1024/224.
                    if i >= 1:
                        crop_dims.append([min_x,min_y,max_x,max_y])
                        image_dict['dims'].append([min_x,max_x,min_y,max_y])
                    else :
                        crop_dims = [min_x,min_y,max_x,max_y]
                        image_dict['dims'].append([min_x,max_x,min_y,max_y])
                    
                    outs = []
                    for idx in range(len(raw_grasp_labels[i])):
                        
                        x1 = raw_grasp_labels[i][idx][0] = (raw_grasp_labels[i][idx][0] - (min_x) + self.border_size) * 224/(max_x-min_x + 2 * self.border_size)
                        y1 = raw_grasp_labels[i][idx][1] = (raw_grasp_labels[i][idx][1] -  (min_y) + self.border_size) * 224/(max_y - min_y + self.border_size * 2)
                        
                        w1 = raw_grasp_labels[i][idx][3] = (raw_grasp_labels[i][idx][3]) * 224/(max_x-min_x + 2 * self.border_size)
                        h1 = raw_grasp_labels[i][idx][4] = (raw_grasp_labels[i][idx][4]) * 224/(max_y-min_y + 2 * self.border_size)
                        #import pdb; pdb.set_trace()
                        cur = [x1.item(),y1.item(),raw_grasp_labels[i][idx][2],w1.item(),h1.item()]
                        outs.append(cur)
                        
                    out_grasps.append(outs)
                    grasps, raw_labels, grippers_points = self.create_grasp_rectangle(raw_grasp_labels[i])
                    gripper_labels.append(grippers_points.copy())
                    grasping_labels.append(grasps)
            
            #import pdb; pdb.set_trace()            
            grasp_labels_vis = raw_grasp_labels.copy()
            if vis == True :
                    for idx,_ in enumerate(grasp_txts):
                        
                        ##rescale midpoint for plot on original image
                        if self.crop:
                            for ix in range(len(grasp_labels_vis[idx])):
                                grasp_labels_vis[idx][ix][0] = grasp_labels_vis[idx][ix][0] * (image_dict['dims'][idx][1] - image_dict['dims'][idx][0] + 2 * self.border_size) /224.
                                grasp_labels_vis[idx][ix][1] = grasp_labels_vis[idx][ix][1] * (image_dict['dims'][idx][3] - image_dict['dims'][idx][2] + 2 * self.border_size) /224.
                                grasp_labels_vis[idx][ix][3] = grasp_labels_vis[idx][ix][3] * (image_dict['dims'][idx][1] - image_dict['dims'][idx][0] + 2 * self.border_size) /224.
                                grasp_labels_vis[idx][ix][4] = grasp_labels_vis[idx][ix][4] * (image_dict['dims'][idx][3] - image_dict['dims'][idx][2] + 2 * self.border_size) /224.
                        
                        grasp_vis, _, gripper_vis = self.create_grasp_rectangle(grasp_labels_vis[idx])
                        if self.crop : 
                            self.visualize(imgs[idx],grasp_vis,gripper_vis,store_dir,image_dict['dims'][idx])
                        else : 
                            self.visualize(imgs[idx],grasp_vis,gripper_vis,store_dir)
                        self.img_cnt += 1
            #import pdb; pdb.set_trace()
            for im_idx in range(len(img_paths)):
                cur_img_pth = img_paths[im_idx]
                cur_mask = masks[im_idx]
                cur_grasps = gripper_labels[im_idx] #two side points 
                gknet_labels = out_grasps[im_idx] #grasp line with x,y,angle,w
                if self.crop : 
                    cur_crop_dims = crop_dims[im_idx]
                    
                
                for label_num in range(len(gknet_labels)):
                    cur_dict = {}
                    cur_dict['img_path'] = cur_img_pth
                    cur_dict['mask_path'] = cur_mask
                    cur_dict['two_point_labels'] = cur_grasps[label_num]
                    cur_dict['gknet_labels'] = gknet_labels[label_num][:4]
                    if self.crop : 
                        cur_dict['crop_dims'] = cur_crop_dims
                        
                    self.items.append(cur_dict)
    
    def distance(self,ax, ay, bx, by):
        return math.sqrt((by - ay)**2 + (bx - ax)**2)
    
    def rotate(self,origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy
    
    def rotated_about(self,ax, ay, bx, by, angle):
        radius = self.distance(ax,ay,bx,by)
        angle += math.atan2(ay-by, ax-bx)
        
        #import pdb; pdb.set_trace()
        try:
            return [
                round(bx.item() + radius * math.cos(angle)),
                round(by.item() + radius * math.sin(angle))
            ]
        except : 
            return [
                round(bx + radius * math.cos(angle)),
                round(by + radius * math.sin(angle))
            ]
            
    def rotate_points(self,center,points,theta):
        pts_new = []
        cx,cy = center
        for pt in points:
            x,y = pt
            tempX = x - cx
            tempY = y - cy
            
            rotatedX = tempX*np.cos(theta) - tempY*np.sin(theta);
            rotatedY = tempX*np.sin(theta) + tempY*np.cos(theta);
        
    def create_grasp_rectangle(self,grasp):
        mids = []
        grasps = []
        centers = []
        gripper_poses = []
        for n,el in enumerate(grasp) : 
            #import pdb; pdb.set_trace()
            #import pdb; pdb.set_trace()
            x,y,angle,w,h = el
            #if [x,y] in mids: deactivate for the plots
            #    continue
            mids.append([x,y])
            tl = (x - w/2, y - h/2)
            bl = (x - w/2, y + h/2)
            tr = (x+ w/2, y - h/2)
            br = (x + w/2, y + h/2)
            #use left and right point representation as in GKNet, choose the midpoints of the vertical lines
            mr = ((br[0] + tr[0]) * 0.5, (br[1] + tr[1]) * 0.5)
            ml = ((bl[0] + tl[0]) * 0.5, (bl[1] + tl[1]) * 0.5)
            
            points = [br,tr,tl,bl]
            grippers = [ml,mr]
            #tl,bl,tr,br = self.rotate_points((x,y),points,angle) 
            #print(len(points))
            square_vertices = [self.rotated_about(pt[0],pt[1], x, y, math.radians(angle)) for pt in points]      
            gripper_points = [self.rotated_about(pt[0],pt[1], x, y, math.radians(angle)) for pt in grippers]
            grasps.append(square_vertices)
            centers.append([x,y,angle,w,h])
            gripper_poses.append(gripper_points)
            #br,tr,tl,bl = square_vertices
        #import pdb; pdb.set_trace()
        return grasps, centers, gripper_poses
    
    def visualize(self,img,grasp,gripper_points,store_dir,dims=None):
        #img = Image.open(img) 
        #import pdb; pdb.set_trace()
        try :
            img = cv2.imread(img)
        except :
            img = img[0]
            img = cv2.imread(img)
            
        if self.crop : 
            #import pdb; pdb.set_trace()
            img = img[int(dims[2]) -self.border_size:int(dims[3]) + self.border_size,int(dims[0]) - self.border_size:int(dims[1]) + self.border_size,:]
        #draw = ImageDraw.Draw(img)
        for n,el in enumerate(grasp) : 
            br,tr,tl,bl = el
            point_left, point_right = gripper_points[n]
            #br[0],br[1] = int(br[0] * w/224.),int(br[1] * h /224.)
            #bl[0],bl[1] = int(bl[0] * w/224.),int(bl[1] * h /224.)
            #tl[0],tl[1] = int(tl[0] * w/224.),int(tl[1] * h /224.)
            #tr[0],tr[1] = int(tr[0] * w/224.),int(tr[1] * h /224.)
            
            #print(len(square_vertices))    
            color1 = (list(np.random.choice(range(256), size=3)))  
            color =  [int(color1[0]), int(color1[1]), int(color1[2])]  
            img = cv2.circle(img,(int(point_left[0]),int(point_left[1])),2,(255,0,0),thickness=2)
            img = cv2.circle(img,(int(point_right[0]),int(point_right[1])),2,(0,255,0),thickness=2)
            img = cv2.line(img,tl,bl,color,thickness=2)
            img = cv2.line(img,tr,br,color,thickness=2)
            img = cv2.line(img,br,bl,color,thickness=2)
            img = cv2.line(img,tr,tl,color,thickness=2)
        
        cv2.imwrite(store_dir + str(self.img_cnt) + '.png',img)
        self.img_cnt += 1
            
    def visualize_imgs(self,img,grasp,grasp_transformed,grasp_ref,img_ref,dst,centers_new,new_pts,store_dir,dims=None,dims_ref=None):
        #img = Image.open(img) 
        #import pdb; pdb.set_trace()
        #img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = cv2.imread(img)
        
        #import pdb; pdb.set_trace()
        #grasp[ix][0] = grasp_labels_vis[idx][ix][0] * (image_dict['dims'][idx][1] - image_dict['dims'][idx][0] + 2 * self.border_size) /224.
        #grasp_labels_vis[idx][ix][1] = grasp_labels_vis[idx][ix][1] * (image_dict['dims'][idx][3] - image_dict['dims'][idx][2] + 2 * self.border_size) /224.
        #grasp_labels_vis[idx][ix][3] = grasp_labels_vis[idx][ix][3] * (image_dict['dims'][idx][1] - image_dict['dims'][idx][0] + 2 * self.border_size) /224.
        #grasp_labels_vis[idx][ix][4] = grasp_labels_vis[idx][ix][4] * (image_dict['dims'][idx][3] - image_dict['dims'][idx][2] + 2 * self.border_size) /224.
        grasps = grasp[0]
        all_grasps = []
        for g in grasps : 
            
            if self.crop : 
                g[0] = g[0] * (dims[1] - dims[0] + 2 * self.border_size) /224.
                g[1] = g[1] * (dims[3] - dims[2] + 2 * self.border_size) /224.
                g[3] = g[3] * (dims[1] - dims[0] + 2 * self.border_size) /224.
                g[4] = g[4] * (dims[3] - dims[2] + 2 * self.border_size) /224.
            
                
                
            all_grasps.append(g)
            
        grasp_vis, _, grasp_pts = self.create_grasp_rectangle(all_grasps)
        
        ref_grasps = []
        for g in grasp_ref:
            if self.crop : 
                g[0] = g[0] * (dims_ref[1] - dims_ref[0] + 2 * self.border_size) /224.
                g[1] = g[1] * (dims_ref[3] - dims_ref[2] + 2 * self.border_size) /224.
                g[3] = g[3] * (dims_ref[1] - dims_ref[0] + 2 * self.border_size) /224.
                g[4] = g[4] * (dims_ref[3] - dims_ref[2] + 2 * self.border_size) /224.
            
            ref_grasps.append(g)
        grasp_vis_ref, _, grasp_pts_ref = self.create_grasp_rectangle(ref_grasps)
        
        try : 
            img3 = cv2.imread(img_ref)
        except:
            img3 = cv2.imread(img_ref[0])
        if self.crop :
            img = img[int(dims[2]) -self.border_size:int(dims[3]) + self.border_size,int(dims[0]) - self.border_size:int(dims[1]) + self.border_size,:]
            img3 = img3[int(dims_ref[2]) -self.border_size:int(dims_ref[3]) + self.border_size,int(dims_ref[0]) - self.border_size:int(dims_ref[1]) + self.border_size,:]
        img2 = img.copy()
        #draw = ImageDraw.Draw(img)
        for n,el in enumerate(grasp_vis) : 
            #import pdb; pdb.set_trace()
            
            br,tr,tl,bl = el
            point_left, point_right = grasp_pts[n]
            #import pdb; pdb.set_trace()
            if not self.crop :
                point_left = [int(i * 1024/224) for i in point_left]
                point_right = [int(i * 1024/224) for i in point_right]
            #br = [int(i * 1024/224.) for i in br]
            #tr = [int(i * 1024/224.) for i in tr]
            #bl = [int(i * 1024/224.) for i in bl]
            #tl = [int(i * 1024/224.) for i in tl]
            
            #print(len(square_vertices))    
            color1 = (list(np.random.choice(range(256), size=3)))
            color =[int(color1[0]), int(color1[1]), int(color1[2])] 
            img = cv2.circle(img,(int(point_left[0]),int(point_left[1])),2,(255,0,0),thickness=2)
            img = cv2.circle(img,(int(point_right[0]),int(point_right[1])),2,(0,255,0),thickness=2) 
            img = cv2.line(img,(int(point_left[0]),int(point_left[1])),(int(point_right[0]),int(point_right[1])),(0,0,255),thickness=2)
            #img = cv2.line(img,tl,bl,color,thickness=2)
            #img = cv2.line(img,tr,br,color,thickness=2)
            #img = cv2.line(img,br,bl,color,thickness=2)
            #img = cv2.line(img,tr,tl,color,thickness=2)
        
        cv2.imwrite(store_dir + str(self.img_cnt) + 'gt.png',img)
        
        '''
        for n,el in enumerate(grasp_transformed) : 
            #import pdb; pdb.set_trace()
            br,tr,tl,bl = el
            br = [int(i ) for i in br]
            tr = [int(i ) for i in tr]
            bl = [int(i ) for i in bl]
            tl = [int(i ) for i in tl]
            
            br = [br[1],br[0]]
            tr = [tr[1],tr[0]]
            bl = [bl[1],bl[0]] 
            tl = [tl[1],tl[0]]

            color1 = (list(np.random.choice(range(256), size=3)))  
            color =[int(color1[0]), int(color1[1]), int(color1[2])]  
            img2 = cv2.line(img2,tl,bl,color,thickness=2)
            img2 = cv2.line(img2,tr,br,color,thickness=2)
            img2 = cv2.line(img2,br,bl,color,thickness=2)
            img2 = cv2.line(img2,tr,tl,color,thickness=2)
            #img2 = cv2.circle(img2,(br[1],br[0]), 2, (255,0,0), -1)
        '''
        for n, el in enumerate(new_pts):
            ptl, ptr = el[0], el[1]
            #dims = [i.item() for i in dims]
            if self.crop : 
                ptl[0] = ptl[0] * (dims[3] - dims[2] + 2 * self.border_size) /224.
                ptr[0] = ptr[0] * (dims[3] - dims[2] + 2 * self.border_size) /224.
                ptl[1] = ptl[1] * (dims[1] - dims[0] + 2 * self.border_size) /224.
                ptr[1] = ptr[1] * (dims[1] - dims[0] + 2 * self.border_size) /224.
            else : 
                ptl[0] = ptl[0] * 1024/224.
                ptr[0] = ptr[0] * 1024/224.
                ptr[1] = ptr[1] * 1024/224.
                ptl[1] = ptl[1] * 1024/224.
            
                
            ptl = [ptl[1],ptl[0]]
            ptr = [ptr[1],ptr[0]]
            #import pdb; pdb.set_trace()
            img2 = cv2.circle(img2,(int(ptl[0]),int(ptl[1])), 2, (255,0,0), -1)
            img2 = cv2.circle(img2,(int(ptr[0]),int(ptr[1])), 2, (0,255,0), -1)
            img2 = cv2.line(img2,(int(ptl[0]),int(ptl[1])),(int(ptr[0]),int(ptr[1])),(0,0,255),thickness=2)
            
            
            
            #import pdb; pdb.set_trace()
        if self.crop :
            for pred in dst[0]:
                x,y = pred
                img2 = cv2.circle(img2,(int(y * (dims[1] - dims[0] + 2 * self.border_size) /224.),int(x * (dims[3] - dims[2] + 2 * self.border_size) /224.)), 2, (0,0,255), -1)
            
            for center in centers_new[:,0,:]:
                x,y = center
                img2 = cv2.circle(img2,(int(y * (dims[1] - dims[0] + 2 * self.border_size) /224.),int(x * (dims[3] - dims[2] + 2 * self.border_size) /224.)), 2, (255,255,0), -1)
            
        
        cv2.imwrite(store_dir + str(self.img_cnt) + '_transformed.png',img2)
        
        for n,el in enumerate(grasp_vis_ref) : 
            #import pdb; pdb.set_trace()
            br,tr,tl,bl = el
            point_left, point_right = grasp_pts_ref[n]
            #import pdb; pdb.set_trace()
            if not self.crop :
                point_left = [int(i * 1024/224) for i in point_left]
                point_right = [int(i * 1024/224) for i in point_right]
            #br = [int(i * 1024/224.) for i in br]
            #tr = [int(i * 1024/224.) for i in tr]
            #bl = [int(i * 1024/224.) for i in bl]
            #tl = [int(i * 1024/224.) for i in tl]
            
            #print(len(square_vertices))    
            color1 = (list(np.random.choice(range(256), size=3)))
            color =[int(color1[0]), int(color1[1]), int(color1[2])] 
            img3 = cv2.circle(img3,(int(point_left[0]),int(point_left[1])),2,(255,0,0),thickness=2)
            img3 = cv2.circle(img3,(int(point_right[0]),int(point_right[1])),2,(0,255,0),thickness=2) 
            img3 = cv2.line(img3,(int(point_left[0]),int(point_left[1])),(int(point_right[0]),int(point_right[1])),(0,0,255),thickness=2)
        
        
        cv2.imwrite(store_dir + str(self.img_cnt) + 'ref.png',img3)
        
        self.img_cnt += 1 

    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self,index):
        img = self.items[index]['img_path']
        mask = self.items[index]['mask_path']
        points_grasps = self.items[index]['two_point_labels']
        gknet_label = self.items[index]['gknet_labels']
        
        '''
        if self.crop : 
            ref_dims = self.items[index]['ref_dims']
            
        
        if self.crop : 
            crop_ref = self.items[index]['crop_ref']
        '''
        
        img_raw = Image.open(img) 
        mask = Image.open(mask)
        '''
        if self.crop :
            left,top,right,bottom = crop_ref
            left = left - self.border_size
            top = top - self.border_size 
            right = right + self.border_size
            bottom = bottom + self.border_size
            img_ref = img_ref.crop((left.item(), top.item(), right.item(), bottom.item()))
            #img_ref = img_ref.crop((top.item(), left.item(), bottom.item(), right.item()))
            mask_ref = mask_ref.crop((left.item(), top.item(), right.item(), bottom.item()))
            #mask_ref = mask_ref.crop((top.item(), left.item(), bottom.item(), right.item()))
        '''
        
        img = self.image_transform(img_raw)
        mask = self.transform_tensor(mask)
        
        
        angle = 10 
        augmented_img = augment_image(img,angle)
        augmented_gknet_label = gknet_label.copy()  
        augmented_gknet_label[2] = augmented_gknet_label[2] - angle #just change the angle of the label here
        x,y = self.rotated_about(augmented_gknet_label[0],augmented_gknet_label[1],224/2.,224/2.,math.radians(-angle))
        augmented_gknet_label[0] = x
        augmented_gknet_label[1] = y
        
        '''
        if self.crop == False :
            grasps_ref = grasps_ref * 224/1024
        '''
            
        data_dict = {}
        data_dict['img_grasp'] = torch.tensor(gknet_label) 
        data_dict['img'] = img
        data_dict['img_augmented'] = augmented_img  
        data_dict['img_augmented_grasp'] = torch.tensor(augmented_gknet_label) 
        
        return data_dict