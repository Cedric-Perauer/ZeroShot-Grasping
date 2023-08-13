from torch.utils.data import Dataset
import torch
import numpy as np
from torchvision import transforms

import os
from PIL import Image
import random
from utils import get_grasp, get_augmented_angles, get_transform_mask

jacquard_root  = os.getcwd()+ r'/data/categories_mlp_1training_sample/categories/'


class JacquardSamples(Dataset):
    '''
    very simple dataloader to test some stuff on Jacquard samples dataset
    '''

    def __init__(self, dataset_root="",
                 image_transform=None,
                 num_targets=2, crop=False, overfit=False, img_size=224,idx=0):
        self.img_size = img_size
        self.dataset_root = jacquard_root + dataset_root
        self.image_transform = image_transform
        self.mask_transform = get_transform_mask()
        self.classes = os.listdir(self.dataset_root)
        print('classes', self.classes)
        self.image_norm_mean = (0.485, 0.456, 0.406)
        self.image_norm_std = (0.229, 0.224, 0.225)
        self.crop = crop
        self.overfit = overfit
        self.mask_paths = []
        self.image_paths = []
        self.grasp_txts = []
        self.nums = []
        for num_c, cat in enumerate(self.classes):
            if os.path.isdir(self.dataset_root + cat) == False :
                continue

            fs = os.listdir(self.dataset_root + cat)

            imgs = [self.dataset_root + cat + "/" + i for i in fs if i.endswith('.jpg') or i.endswith('.png')]
            #grasp_txts = [self.dataset_root + cat + "/" + i for i in fs if i.endswith('.txt')]
            #grasp_txts = [i.split('/')[-1] for i in grasp_txts]
            imgs = [i.split('/')[-1] for i in imgs if 'RGB' in i]

            imgs = sorted(imgs, key=lambda x: int(x.split('_')[0]))
            #grasp_txts = sorted(grasp_txts, key=lambda x: int(x.split('_')[0]))
            imgs = [self.dataset_root + cat + "/" + i for i in imgs]
            img_masks = [i.replace('RGB', 'mask') for i in imgs]
            grasp_txts = [i.replace('RGB', 'grasps').replace('.png','.txt') for i in imgs]
            if len(grasp_txts) != len(imgs) :
                raise Exception("Number of images and grasp files do not match at object", cat)
            self.mask_paths.extend(img_masks)
            self.image_paths.extend(imgs)
            self.grasp_txts.extend(grasp_txts)
            self.nums.append([len(imgs)])
        
        #select_idx = np.sum(np.array(self.nums)[:idx])
        select_idx = len(self.image_paths)
        #print('select_idx', select_idx)
        #self.mask_paths = np.array(self.mask_paths)[:select_idx]
        #self.image_paths = np.array(self.image_paths)[:select_idx]
        #self.grasp_txts = np.array(self.grasp_txts)[:select_idx]
        if (len(self.image_paths) == len(self.grasp_txts) == len(self.mask_paths)) == False:
            raise Exception("Number of images and grasp files do not match")


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        data_dict = {}
        img_raw = Image.open(self.image_paths[index])
        mask = Image.open(self.mask_paths[index])
        img = self.image_transform(img_raw)
        mask = self.mask_transform(mask)

        points_grasps, gknet_labels, corners = get_grasp(self.grasp_txts[index], self.img_size, self.crop)
        points_grasps = torch.tensor(points_grasps).squeeze()
        gknet_labels = torch.tensor(gknet_labels).squeeze()
        data_dict['points_grasp'] = torch.tensor(points_grasps)
        data_dict['img'] = img
        data_dict['mask'] = mask
        data_dict['corners'] = torch.tensor(corners)
        data_dict['raw'] = torch.tensor(gknet_labels)
        data_dict['angle'] = []
        data_dict['img_grasp'] = []
        data_dict['points_grasp_augmented'] = []
        data_dict['points_grasp_augmented2'] = []
        data_dict['img_augmented_grasp'] = []
        data_dict['img_augmented_grasp2'] = []
        data_dict['height'] = []
        for i in range(gknet_labels.shape[0]):
            angle = random.randint(-180, 180)
            data_dict["angle"].append(angle)
            augmented_gknet_label, augmented_gknet_label2, gknet_label, pg_aug, pg_aug2 = get_augmented_angles(gknet_labels[i].numpy().copy(), angle, self.img_size)
            data_dict['img_grasp'].append(gknet_label)
            data_dict['points_grasp_augmented'].append(pg_aug)
            data_dict['points_grasp_augmented2'].append(pg_aug2)
            data_dict['img_augmented_grasp'].append(augmented_gknet_label)
            data_dict['img_augmented_grasp2'].append(augmented_gknet_label2)
            data_dict['height'].append(gknet_label[-1])
        return data_dict




