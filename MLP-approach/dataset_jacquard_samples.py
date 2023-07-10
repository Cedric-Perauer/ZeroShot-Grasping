from torch.utils.data import Dataset
import torch
import numpy as np
from torchvision import transforms

import os
from PIL import Image
import random
from utils import get_grasp, get_augmented_angles, get_transform_mask

jacquard_root  =os.getcwd()+ r'/data/Bottle/'


class JacquardSamples(Dataset):
    '''
    very simple dataloader to test some stuff on Jacquard samples dataset
    '''

    def __init__(self, dataset_root=jacquard_root,
                 image_transform=None,
                 num_targets=2, crop=False, overfit=False, img_size=224):
        self.img_size = img_size
        self.dataset_root = dataset_root
        self.image_transform = image_transform
        self.mask_transform = get_transform_mask()
        self.classes = os.listdir(dataset_root)
        self.image_norm_mean = (0.485, 0.456, 0.406)
        self.image_norm_std = (0.229, 0.224, 0.225)
        self.crop = crop
        self.overfit = overfit
        for num_c, cat in enumerate(self.classes):
            if os.path.isdir(self.dataset_root + cat) == False :
                continue
            fs = os.listdir(self.dataset_root + cat)

            imgs = [self.dataset_root + cat + "/" + i for i in fs if i.endswith('.jpg') or i.endswith('.png')]
            grasp_txts = [self.dataset_root + cat + "/" + i for i in fs if i.endswith('.txt')]
            grasp_txts = [i.split('/')[-1] for i in grasp_txts]
            imgs = [i.split('/')[-1] for i in imgs if 'RGB' in i]

            imgs = sorted(imgs, key=lambda x: int(x.split('_')[0]))
            grasp_txts = sorted(grasp_txts, key=lambda x: int(x.split('_')[0]))
            imgs = [self.dataset_root + cat + "/" + i for i in imgs]
            img_masks = [i.replace('RGB', 'mask') for i in imgs]
            grasp_txts = [self.dataset_root + cat + "/" + i for i in grasp_txts]
        self.mask_paths = img_masks
        self.image_paths = imgs
        self.grasp_txts = grasp_txts


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        data_dict = {}
        img_raw = Image.open(self.image_paths[index])
        mask = Image.open(self.mask_paths[index])
        img = self.image_transform(img_raw)
        mask = self.mask_transform(mask)

        points_grasps, gknet_labels = get_grasp(self.grasp_txts[index], self.img_size, self.crop)
        points_grasps = torch.tensor(points_grasps).squeeze()
        gknet_labels = torch.tensor(gknet_labels).squeeze()
        data_dict['points_grasp'] = torch.tensor(points_grasps)
        data_dict['img'] = img
        data_dict['mask'] = mask
        data_dict['angle'] = []
        data_dict['img_grasp'] = []
        data_dict['points_grasp_augmented'] = []
        data_dict['points_grasp_augmented2'] = []
        data_dict['img_augmented_grasp'] = []
        data_dict['img_augmented_grasp2'] = []
        for i in range(gknet_labels.shape[0]):
            angle = random.randint(-180, 180)
            data_dict["angle"].append(angle)
            augmented_gknet_label, augmented_gknet_label2, gknet_label, pg_aug, pg_aug2 = get_augmented_angles(gknet_labels[i].numpy().copy(), angle, self.img_size)
            data_dict['img_grasp'].append(gknet_label)
            data_dict['points_grasp_augmented'].append(pg_aug)
            data_dict['points_grasp_augmented2'].append(pg_aug2)
            data_dict['img_augmented_grasp'].append(augmented_gknet_label)
            data_dict['img_augmented_grasp2'].append(augmented_gknet_label2)
        return data_dict




