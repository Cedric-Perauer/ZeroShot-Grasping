# Co3D
from zsp.datasets.co3d_dataset import Co3dDataset
#from zsp.datasets.co3d_dataset_fix import Co3dDataset

# ML
from torch.utils.data import Dataset
import torch
import numpy as np
import random
from torchvision import transforms
import math
import cv2
import shutil
# I/O
import os
import json
from PIL import Image, ImageDraw

# Typing
from typing import List

co3d_root = os.path.expanduser("~") + '/ZeroShot-Grasping/zero-shot-pose/co3d/'
label_dir = os.path.expanduser("~") + '/ZeroShot-Grasping/zero-shot-pose/data/class_labels/'
jacquard_root  = os.path.expanduser("~") + '/ZeroShot-Grasping/zero-shot-pose/Jacquard/Samples/'

class TestDataset(Dataset):
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
                with open(txt,'r') as f:
                    lines = f.readlines()
                    for l in lines :
                        split = l.split(';')
                        x,y,angle,w,h = split 
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
            ## crop the images to only the are where the object is present plus some boarder area
            if self.crop :
                
                ## for target masks
                cur_dict['crop_target'] = []
                for i in range(0,len(masks)):
                    mask_ref = self.transform_tensor(Image.open(masks[i]))
                    idcs = (mask_ref != 0).nonzero(as_tuple=False)[:,1:3]
                    max_y,min_y,max_x,min_x = idcs[:,0].max()* 1024/224.,idcs[:,0].min()* 1024/224.,idcs[:,1].max()* 1024/224.,idcs[:,1].min()* 1024/224.
                    if i >= 1:
                        cur_dict['crop_target'].append([min_x,min_y,max_x,max_y])
                        image_dict['dims'].append([min_x,max_x,min_y,max_y])
                    else :
                        cur_dict['crop_ref'] = [min_x,min_y,max_x,max_y]
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
            cur_dict['ref_img_path'] = img_paths[0]
            cur_dict['ref_raw_labels'] = out_grasps[0]
            cur_dict['target_img_path'] = img_paths[1:]
            cur_dict['ref_gripper_pts'] = gripper_labels[0]
            cur_dict['target_gripper_pts'] = gripper_labels[1:]
            cur_dict['ref_image_rgb'] = imgs[0]
            cur_dict['ref_image_mask'] = masks[0]
            cur_dict['target_raw_labels'] = out_grasps[1:]
            cur_dict['target_grasps'] = grasping_labels[1:]
            cur_dict['ref_grasps'] = grasping_labels[0]
            if self.crop : 
                cur_dict['ref_dims'] = image_dict['dims'][0]
                cur_dict['target_dims'] = image_dict['dims'][1:]
            
            
            
            up = num_targets + 1 if len(imgs) > num_targets else len(imgs) + 1
            cur_dict['target_images_masks'] = masks[1:up]
            cur_dict['target_images_rgb'] = imgs[1:up]
            ##tbd : get grasp labels 
            
            self.items.append(cur_dict)
    
    def distance(self,ax, ay, bx, by):
        return math.sqrt((by - ay)**2 + (bx - ax)**2)
    
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
            if [x,y] in mids:
                continue
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
        
        try : 
            grasp_vis_ref, _, grasp_pts_ref = self.create_grasp_rectangle(ref_grasps)
        except : 
            import pdb; pdb.set_trace()
        
        try : 
            img3 = cv2.imread(img_ref)
        except:
            img3 = cv2.imread(img_ref[0])
        if self.crop :
            img = img[int(dims[2]) -self.border_size:int(dims[3]) + self.border_size,int(dims[0]) - self.border_size:int(dims[1]) + self.border_size,:]
            img3 = img3[int(dims_ref[2]) -self.border_size:int(dims_ref[3]) + self.border_size,int(dims_ref[0]) - self.border_size:int(dims_ref[1]) + self.border_size,:]
        img2 = img3.copy()
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
        img_ref = self.items[index]['ref_image_rgb']
        target_imgs = self.items[index]['target_images_rgb']
        mask_ref = self.items[index]['ref_image_mask']
        mask_target = self.items[index]['target_images_masks']
        grasps_target = self.items[index]['target_grasps']
        grasps_ref = self.items[index]['ref_grasps']
        ref_path = self.items[index]['ref_img_path']
        target_path = self.items[index]['target_img_path']
        target_raw_labels = self.items[index]['target_raw_labels']
        ref_raw_labels = self.items[index]['ref_raw_labels']
        if self.crop : 
            ref_dims = self.items[index]['ref_dims']
            target_dims = self.items[index]['target_dims']
        ref_gripper_pts = self.items[index]['ref_gripper_pts']
        target_gripper_pts = self.items[index]['target_gripper_pts']
        crop_ref,crop_target = None,None
        
        if self.crop : 
            crop_ref = self.items[index]['crop_ref']
            crop_target = self.items[index]['crop_target']

        img_ref = Image.open(img_ref) 
        mask_ref = Image.open(mask_ref)
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
        
        img_ref = self.image_transform(img_ref)
        mask_ref = self.transform_tensor(mask_ref)
        mask_targets = [Image.open(i) for i in mask_target]
        target_imgss = [Image.open(i) for i in target_imgs]
        if self.crop :
            mask_target, target_imgs = [],[]
            for i in range(len(mask_targets)):
                left,top,right,bottom = crop_target[i]
                left = left - self.border_size
                top = top - self.border_size 
                right = right + self.border_size
                bottom = bottom + self.border_size
                mask_crop = mask_targets[i]
                img_crop = target_imgss[i]
                mask_crop = mask_targets[i].crop((left.item(), top.item(), right.item(), bottom.item()))
                img_crop = target_imgss[i].crop((left.item(),  top.item(), right.item(), bottom.item()))
                mask_target.append(self.transform_tensor(mask_crop))
                target_imgs.append(self.image_transform(img_crop))
             
        else : 
            mask_target = [self.transform_tensor(i) for i in mask_targets]
            target_imgs = [self.image_transform(i) for i in target_imgss]
        
        target_raw_labels = [torch.tensor(i) for i in target_raw_labels]
        grasps_target = [torch.tensor(i) for i in grasps_target]
        ref_raw_labels = torch.tensor(ref_raw_labels) 
        grasps_ref = torch.tensor(grasps_ref) 
        if self.crop == False :
            grasps_target = [i * 224/1024 for i in grasps_target]
            grasps_ref = grasps_ref * 224/1024
            
        target_imgs = torch.stack(target_imgs)

        if self.crop : 
            return img_ref, target_imgs, mask_ref,mask_target, grasps_ref,grasps_target, \
                    ref_path, target_path, target_raw_labels, ref_raw_labels, ref_dims, target_dims, \
                        ref_gripper_pts, target_gripper_pts
        else : 
            return img_ref, target_imgs, mask_ref,mask_target, grasps_ref,grasps_target, \
                    ref_path, target_path, target_raw_labels, ref_raw_labels, \
                        ref_gripper_pts, target_gripper_pts
        
        



class Co3DPoseDataset(Dataset):

    def __init__(self, dataset_root=co3d_root,
                 categories: List[str] = ['bicycle'],
                 num_samples_per_class: int = 5,
                 target_frames_sampling_mode: str = 'uniform',
                 num_frames_in_target_seq: int = 5,
                 label_dir: str = label_dir,
                 determ_eval_root: str = None,
                 image_transform=None):

        # Get all candidate sequences for a class
        candidate_seqs = {}
        categories = ['backpack']
        for cat in categories:

            candidate_seqs[cat] = []
            cat_dir = os.path.join(label_dir, cat)

            for f in os.listdir(cat_dir):
                candidate_seqs[cat].append(f.rstrip('.json'))

        # Dict of datasets, one for each class
        self.classes = categories
        self.all_datasets = {
            cls: self.get_base_dataset(dataset_root, cls)
            for cls in self.classes
        }

        # Stores which sequences contain which Frame IDs
        self.labelled_seqs_to_frames = {cls: self.construct_labelled_seq_dict(cls, candidate_seqs[cls])
                              for cls in self.classes}

        # Stores all the sequences which have labels
        self.labelled_seqs = candidate_seqs

        # Computes and stores samples of (reference_frame, [target_frame_0...target_frame_N]) pairs
        # Stored in format
        # sample = {
        #         'class': class,
        #         'reference_seq_name': root_seq_name,
        #         'reference_frame_id': root_frame_id,
        #         'target_seq_name': target_seq_name,
        #         'all_target_seq_id': target_frame_ids
        #         }
        if determ_eval_root:
            if len(categories) != 1:
                raise ValueError('Can only use determ_eval_path when a single category is being loaded')
            determ_eval_path = os.path.join(determ_eval_root, f'{categories[0]}.pt')
            if os.path.exists(determ_eval_path):
                self.samples = torch.load(determ_eval_path)
                if len(self.samples) < num_samples_per_class:
                    raise ValueError(
                        f'Only {len(self.samples)} samples in the determ set, but asked for {num_samples_per_class}')
                self.samples = self.samples[:num_samples_per_class]
                print(f"Successfully loaded determ eval set for category {categories[0]}, kept {len(self.samples)} samples")
            else:
                self.samples = self.sample_instances(num_samples_per_class=num_samples_per_class,
                                                    num_frames_in_target_seq=num_frames_in_target_seq,
                                                    target_frame_sampling_mode=target_frames_sampling_mode)
        else:
            self.samples = self.sample_instances(num_samples_per_class=num_samples_per_class,
                                                num_frames_in_target_seq=num_frames_in_target_seq,
                                                target_frame_sampling_mode=target_frames_sampling_mode)
        # Load ground truth coordinate frames
        self.seq_world_coords = self.load_seq_world_coords(label_dir)

        # Transform on image data
        self.image_transform = image_transform

    def get_base_dataset(self, dataset_root, cls):
        return Co3dDataset(frame_annotations_file=os.path.join(dataset_root, cls, 'frame_annotations.jgz'),
                                sequence_annotations_file=os.path.join(dataset_root, cls, 'sequence_annotations.jgz'),
                                # subset_lists_file=os.path.join(dataset_root, cls, 'set_lists.json'),
                                dataset_root=dataset_root,
                                box_crop=True,
                                box_crop_context=0.1,
                                image_height=None,      # Doesn't resize
                                image_width=None,       # Doesn't resize
                                load_point_clouds=True
                                )
        
    def sample_instances(self, num_samples_per_class=1,
                    num_frames_in_target_seq=5, target_frame_sampling_mode='uniform'):

        """
        For all categories, compare 'num_samples_per_class' sequences
        Choose sequence as the root and randomly sample a FrameID
        Sample 'num_frames_in_target_seq' frame IDs from the other frame
        Two sampling modes for the target frames:
            * 'uniform': Randomly sample a first index then uniformly sample the rest
            * 'random': Randomly sample all target frames
        Store all of this in a list of dicts
        """

        def custom_range(n, end, start=0):
            return list(range(start, n)) + list(range(n+1, end))

        np.random.seed(0)
        random.seed(0)

        all_samples = []

        for cls in self.classes:

            # First, sample which pairs of sequences we are going to compare
            labelled_seqs = self.labelled_seqs[cls]
            labelled_seqs = sorted(labelled_seqs)
            num_seqs_in_cls = len(labelled_seqs)
            seq_pairs = []

            root_seq_id = 0
            num_samples_in_this_class_so_far = 0
            while num_samples_in_this_class_so_far < num_samples_per_class:

                root_seq_name = labelled_seqs[root_seq_id]
                target_seq_id = random.choice(custom_range(root_seq_id, num_seqs_in_cls, start=0))
                target_seq_name = labelled_seqs[target_seq_id]
                seq_pairs.append((root_seq_name, target_seq_name))

                num_samples_in_this_class_so_far += 1
                root_seq_id += 1
                root_seq_id = root_seq_id % num_seqs_in_cls

            # Now sample frame IDs from each pair of sequences
            # One frame from the root, and 'num_frames_in_target_seq' in the target
            for root_seq_name, target_seq_name in seq_pairs:

                # Sample frame from root sequence
                root_frame_number = random.choice(
                    range(len(self.labelled_seqs_to_frames[cls][root_seq_name]))
                )

                root_frame_id = self.labelled_seqs_to_frames[cls][root_seq_name][root_frame_number]

                # Sample frames from target sequence
                # Uniform sampling:
                len_current_target_sequence = len(self.labelled_seqs_to_frames[cls][target_seq_name])
                if target_frame_sampling_mode == 'uniform':
                    target_frame_number = random.choice(
                        range(len(self.labelled_seqs_to_frames[cls][target_seq_name]))
                    )
                    target_frame_ids = [
                        self.labelled_seqs_to_frames[cls][target_seq_name][target_frame_number]
                    ]
                    while len(target_frame_ids) < num_frames_in_target_seq:

                        target_frame_number = (target_frame_number + len_current_target_sequence // num_frames_in_target_seq) % len_current_target_sequence
                        target_frame_ids.append(
                            self.labelled_seqs_to_frames[cls][target_seq_name][target_frame_number]
                        )

                # Random sampling:
                elif target_frame_sampling_mode == 'random':

                    target_frame_numbers = random.sample(range(len_current_target_sequence),
                                                         k=num_frames_in_target_seq)
                    target_frame_ids = [
                        self.labelled_seqs_to_frames[cls][target_seq_name][i]
                        for i in target_frame_numbers
                    ]

                else:

                    raise ValueError

                # Construct sample:
                sample = {
                    'class': cls,
                    'reference_seq_name': root_seq_name,
                    'reference_frame_id': root_frame_id,
                    'target_seq_name': target_seq_name,
                    'all_target_id': target_frame_ids
                }

                all_samples.append(sample)

        return all_samples

    def construct_labelled_seq_dict(self, cls, labelled_seq_names):

        ds = self.all_datasets[cls]

        seqs_to_frames = dict([(seq_name, {}) for seq_name in labelled_seq_names])
        for i, frame_ann in enumerate(ds.frame_annots):
            frame_ann = frame_ann['frame_annotation']
            if frame_ann.sequence_name in labelled_seq_names:
                seqs_to_frames[frame_ann.sequence_name][frame_ann.frame_number] = i

        return seqs_to_frames


    def load_seq_world_coords(self, label_dir):

        cat_labelled_dict = {}
        for cat in os.listdir(label_dir):
            cat_labelled_dict[cat] = {}
            cat_dir = os.path.join(label_dir, cat)

            for f in os.listdir(cat_dir):

                label_path = os.path.join(cat_dir, f)
                with open(label_path, 'r') as json_file:
                    seq_trans = json.load(json_file)

                cat_labelled_dict[cat][seq_trans['seq']] = np.array(seq_trans['trans'])

        return cat_labelled_dict


    def process_list_of_frames(self, frames):

        """
        Take a list of frames and return list of:
            Images (PIL or Tensor)
            Scalings
            Depth maps
            Cameras
        """
        images = [co3d_rgb_to_pil(f.image_rgb) for f in frames]

        if self.image_transform is not None:

            images = [self.image_transform(im) for im in images]
            scalings = [
                np.array(f.image_rgb.shape[1:]) / np.array(im.shape[1:]) for im, f in zip(images, frames)
            ]

        else:

            scalings = [
                torch.Tensor([1, 1]) for _ in frames
            ]

        # Get depth maps
        depth_map = [f.depth_map for f in frames]

        # Get cameras
        cameras = [f.camera for f in frames]

        # Pointclouds (just one needed per seq - same for every frame!)
        pcd = frames[0].sequence_point_cloud

        return images, depth_map, scalings, cameras, pcd


    def __getitem__(self, item):

        sample = self.samples[item]
        cls = sample['class']
        ref_seq_name = sample['reference_seq_name']
        target_seq_name = sample['target_seq_name']

        ref_frame_id = sample['reference_frame_id']
        all_target_id = sample['all_target_id']

        base_dataset = self.all_datasets[cls]

        # Get frames
        ref_frame = base_dataset[ref_frame_id]
        all_target_frames = [base_dataset[idx] for idx in all_target_id]

        # Get world coordinate transforms for each seq
        ref_transform = self.seq_world_coords[cls][ref_seq_name]
        target_transform = self.seq_world_coords[cls][target_seq_name]

        # Process ref frame
        ref_image, ref_depth_map, ref_scaling, ref_camera, ref_pcd = (x[0] for x in self.process_list_of_frames([ref_frame]))

        # Process target frames
        all_target_images, all_target_depth_maps, all_target_scalings, \
        all_target_cameras, target_pcd = self.process_list_of_frames(all_target_frames)

        return ref_image, ref_transform, ref_depth_map, ref_scaling, ref_camera,\
               all_target_images, target_transform, all_target_depth_maps, \
                all_target_scalings, all_target_cameras, ref_pcd, target_pcd, item


    def __len__(self):
        return len(self.samples)


class ICPCo3DPoseDataset(Co3DPoseDataset):

    def __init__(self, *args, **kwargs):

        super(ICPCo3DPoseDataset, self).__init__(*args, **kwargs)

    def get_base_dataset(self, dataset_root, cls):
        return Co3dDataset(frame_annotations_file=os.path.join(dataset_root, cls, 'frame_annotations.jgz'),
                                sequence_annotations_file=os.path.join(dataset_root, cls, 'sequence_annotations.jgz'),
                                subset_lists_file=os.path.join(dataset_root, cls, 'set_lists.json'),
                                dataset_root=dataset_root,
                                box_crop=True,
                                box_crop_context=0.1,
                                image_height=224,      # Doesn't resize
                                image_width=224,       # Doesn't resize
                                load_point_clouds=True
                                )

    def process_list_of_frames(self, frames):

        """
        Take a list of frames and return list of:
            Images (PIL or Tensor)
            Scalings
            Depth maps
            Cameras
        """
        images = [co3d_rgb_to_pil(f.image_rgb) for f in frames]

        if self.image_transform is not None:

            images = [self.image_transform(im) for im in images]
            scalings = [
                np.array(f.image_rgb.shape[1:]) / np.array(im.shape[1:]) for im, f in zip(images, frames)
            ]

        else:

            scalings = [
                torch.Tensor([1, 1]) for _ in frames
            ]

        # Get depth maps
        depth_map = [f.depth_map for f in frames]

        # Get cameras
        cameras = [f.camera for f in frames]

        # Pointclouds (just one needed per seq - same for every frame!)
        pcd = frames[0].sequence_point_cloud

        # Foreground probability maps
        fg_probs = [f.fg_probability for f in frames]

        # Original image_rgb - for unprojecting single/few images to pointclouds
        image_rgbs = [f.image_rgb for f in frames]

        return images, depth_map, scalings, cameras, pcd, fg_probs, image_rgbs


    def __getitem__(self, item):

        sample = self.samples[item]
        cls = sample['class']
        ref_seq_name = sample['reference_seq_name']
        target_seq_name = sample['target_seq_name']

        ref_frame_id = sample['reference_frame_id']
        all_target_id = sample['all_target_id']

        base_dataset = self.all_datasets[cls]

        # Get frames
        ref_frame = base_dataset[ref_frame_id]
        all_target_frames = [base_dataset[idx] for idx in all_target_id]

        # Get world coordinate transforms for each seq
        ref_transform = self.seq_world_coords[cls][ref_seq_name]
        target_transform = self.seq_world_coords[cls][target_seq_name]

        # Process ref frame
        ref_image, ref_depth_map, ref_scaling, ref_camera, ref_pcd, ref_fgprob, ref_image_rgb = (
            x[0] for x in self.process_list_of_frames([ref_frame])
        )

        # Process target frames
        all_target_images, all_target_depth_maps, all_target_scalings, all_target_cameras, \
        target_pcd, all_target_fgprobs, all_target_image_rgb = self.process_list_of_frames(
            all_target_frames
        )

        # Now also return additional cameras for the 0th frame from the reference and target sequences
        base_dataset = self.all_datasets[cls]
        ref_sequence_zero_id = self.labelled_seqs_to_frames[cls][ref_seq_name][0]
        target_sequence_zero_id = self.labelled_seqs_to_frames[cls][target_seq_name][0]

        ref_sequence_zero_camera = base_dataset[ref_sequence_zero_id].camera
        target_sequence_zero_camera = base_dataset[target_sequence_zero_id].camera

        return ref_image, ref_transform, ref_depth_map, ref_scaling, ref_camera,\
               all_target_images, target_transform, all_target_depth_maps, \
               all_target_scalings, all_target_cameras, ref_pcd, target_pcd, \
               ref_fgprob, all_target_fgprobs, ref_sequence_zero_camera, target_sequence_zero_camera, \
               ref_image_rgb, all_target_image_rgb, item


class PlottingCo3DPoseDataset(ICPCo3DPoseDataset):

    def __init__(self, *args, **kwargs):

        super(PlottingCo3DPoseDataset, self).__init__(*args, **kwargs)

    def get_base_dataset(self, dataset_root, cls):
        return Co3dDataset(frame_annotations_file=os.path.join(dataset_root, cls, 'frame_annotations.jgz'),
                                sequence_annotations_file=os.path.join(dataset_root, cls, 'sequence_annotations.jgz'),
                                subset_lists_file=os.path.join(dataset_root, cls, 'set_lists.json'),
                                dataset_root=dataset_root,
                                box_crop=True,
                                box_crop_context=0.1,
                                image_height=None,      # Doesn't resize
                                image_width=None,       # Doesn't resize
                                load_point_clouds=True
                                )


def co3d_rgb_to_pil(image_rgb):
    return Image.fromarray((image_rgb.permute(1, 2, 0) * 255).numpy().astype(np.uint8))


def co3d_pose_dataset_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    ref_image, ref_transform, ref_depth_map, ref_scaling, ref_camera, \
    all_target_images, target_transform, all_target_depth_maps, \
    all_target_scalings, all_target_cameras, ref_pcd, target_pcd, item = zip(*batch)

    ref_image = torch.stack(ref_image)
    all_target_images = torch.stack([torch.stack(x) for x in all_target_images])
    ref_transform = torch.as_tensor(np.stack(ref_transform))
    target_transform = torch.as_tensor(np.stack(target_transform))

    idx_into_dataset = torch.as_tensor(item)

    ref_meta_data = {
        'depth_maps': ref_depth_map,
        'scalings': torch.as_tensor(np.stack(ref_scaling)),
        'cameras': ref_camera,
        'pcd': ref_pcd
    }

    target_meta_data = {
        'depth_maps': all_target_depth_maps,
        'scalings': torch.as_tensor(np.stack(all_target_scalings)),
        'cameras': all_target_cameras,
        'pcd': target_pcd
    }

    return (ref_image, all_target_images), (ref_transform, target_transform), (ref_meta_data, target_meta_data), idx_into_dataset



def co3d_icp_pose_dataset_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    ref_image, ref_transform, ref_depth_map, ref_scaling, ref_camera,\
    all_target_images, target_transform, all_target_depth_maps, \
    all_target_scalings, all_target_cameras, ref_pcd, target_pcd, \
    ref_fgprob, all_target_fgprobs, ref_sequence_zero_camera, target_sequence_zero_camera, \
    ref_image_rgb, all_target_image_rgb, item = zip(*batch)

    # Stack images
    ref_image = torch.stack(ref_image)
    all_target_images = torch.stack([torch.stack(x) for x in all_target_images])
    # Stack fullsize, original images
    ref_image_rgb = torch.stack(ref_image_rgb)
    all_target_image_rgb = torch.stack([torch.stack(x) for x in all_target_image_rgb])
    # Stack transforms
    ref_transform = torch.as_tensor(np.stack(ref_transform))
    target_transform = torch.as_tensor(np.stack(target_transform))
    # Stack depth_map
    ref_depth_map = torch.stack(ref_depth_map)
    all_target_depth_maps = torch.stack([torch.stack(x) for x in all_target_depth_maps])
    # Stack fg_probability
    ref_fgprob = torch.stack(ref_fgprob)
    all_target_fgprobs = torch.stack([torch.stack(x) for x in all_target_fgprobs])

    idx_into_dataset = torch.as_tensor(item)

    ref_meta_data = {
        'image_rgb': ref_image_rgb,
        'depth_maps': ref_depth_map,
        'fg_probability': ref_fgprob,
        'scalings': torch.as_tensor(np.stack(ref_scaling)),
        'cameras': ref_camera,
        'pcd': ref_pcd,
        'zero_camera': ref_sequence_zero_camera
    }

    target_meta_data = {
        'image_rgb': all_target_image_rgb,
        'depth_maps': all_target_depth_maps,
        'fg_probability': all_target_fgprobs,
        'scalings': torch.as_tensor(np.stack(all_target_scalings)),
        'cameras': all_target_cameras,
        'pcd': target_pcd,
        'zero_camera': target_sequence_zero_camera
    }

    return (ref_image, all_target_images), (ref_transform, target_transform), (ref_meta_data, target_meta_data),\
           idx_into_dataset

def co3d_plotting_pose_dataset_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    ref_image, ref_transform, ref_depth_map, ref_scaling, ref_camera,\
    all_target_images, target_transform, all_target_depth_maps, \
    all_target_scalings, all_target_cameras, ref_pcd, target_pcd, \
    ref_fgprob, all_target_fgprobs, ref_sequence_zero_camera, target_sequence_zero_camera, \
    ref_image_rgb, all_target_image_rgb, item = zip(*batch)

    # Stack images
    ref_image = torch.stack(ref_image)
    all_target_images = torch.stack([torch.stack(x) for x in all_target_images])
    # Stack transforms
    ref_transform = torch.as_tensor(np.stack(ref_transform))
    target_transform = torch.as_tensor(np.stack(target_transform))
    # NB: don't stack depth_map, fg_probability, image_rgb as these are not
    # all the same size in the PlottingCo3DPoseDataset

    idx_into_dataset = torch.as_tensor(item)

    ref_meta_data = {
        'image_rgb': ref_image_rgb,
        'depth_maps': ref_depth_map,
        'fg_probability': ref_fgprob,
        'scalings': torch.as_tensor(np.stack(ref_scaling)),
        'cameras': ref_camera,
        'pcd': ref_pcd,
        'zero_camera': ref_sequence_zero_camera
    }

    target_meta_data = {
        'image_rgb': all_target_image_rgb,
        'depth_maps': all_target_depth_maps,
        'fg_probability': all_target_fgprobs,
        'scalings': torch.as_tensor(np.stack(all_target_scalings)),
        'cameras': all_target_cameras,
        'pcd': target_pcd,
        'zero_camera': target_sequence_zero_camera
    }

    return (ref_image, all_target_images), (ref_transform, target_transform), (ref_meta_data, target_meta_data),\
           idx_into_dataset

if __name__ == '__main__':
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from torchvision import transforms
    # Image processing
    image_norm_mean = (0.485, 0.456, 0.406)
    image_norm_std = (0.229, 0.224, 0.225)
    image_size = 224    # Image size
    image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_norm_mean, std=image_norm_std)
        ])
    categories = ["backpack", "bicycle", "book", "car", "chair", "hairdryer", "handbag",
              "hydrant", "keyboard", "laptop", "motorcycle", "mouse", "remote", 
              "teddybear", "toaster", "toilet", "toybus", "toyplane", "toytrain", "toytruck"]
    all_samples = {}
    num_samples_per_class = 100
    num_frames_in_target_seq = 50
    determ_eval_root = f'/home/bras3856/Code/Pose/zero-shot-pose/data/determ_eval/200samples_{num_frames_in_target_seq}tgt'
    os.makedirs(determ_eval_root, exist_ok=True)
    for category in categories:
        print(category)
        determ_eval_path = os.path.join(determ_eval_root, f'{category}.pt')
        dataset = Co3DPoseDataset(dataset_root='/data/engs-robot-learning/bras3856/CO3D',
                            categories=[category],
                            num_samples_per_class=num_samples_per_class,
                            target_frames_sampling_mode='uniform',
                            num_frames_in_target_seq=num_frames_in_target_seq,
                            label_dir='/home/bras3856/Code/Pose/zero-shot-pose/data/class_labels',
                            determ_eval_root = determ_eval_root,
                            image_transform=image_transform)
        
        # batch_size = 8
        # dataloader = DataLoader(dataset, num_workers=8,
        #                 batch_size=batch_size, collate_fn=co3d_pose_dataset_collate, shuffle=False)

        # for batch in tqdm(dataloader):
        #     pass
            
        torch.save(dataset.samples, determ_eval_path)
