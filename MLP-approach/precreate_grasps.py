import torch
import numpy as np
from dataset_jacquard_samples import JacquardSamples
from utils import get_transform, augment_image
from bce_model import BCEGraspTransformer
from utils_train import create_correct_false_points, create_correct_false_grasps_mask
import random
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
import os 
import shutil 

'''
this script precreated the false and correct grasps and stores them in .json files for each image 
this is done due to the slow runtime of the sampling and makes the training process simpler and more efficient 
'''


def create_jsons(dataset,  args_train, device):
    output_dir = 'data/preprocess/' + args_train['split']
    if os.path.exists(output_dir) == True :
        shutil.rmtree(output_dir)
        
    os.makedirs(output_dir)
        
    train_loss_running = 0.
    for i in range(len(dataset)):
            data = dataset[i]
            img = data["img"].to(device)
            height = data['height']
            img = torch.permute(img, (0, 2, 1))
            mask = data["mask"].sum().sqrt().to(device)
            obj_mask = data['mask'].to(device)  
            grasp = data["points_grasp"]//14
            path = data["path"]
            #grasp_inv = torch.cat([grasp[:,1,:].unsqueeze(1), grasp[:,0,:].unsqueeze(1)], dim=1)
            #grasp = torch.cat([grasp, grasp_inv], dim=0)
            wrong_far_grasps_right, wrong_far_grasps_left, wrong_mask_grasps_left,wrong_mask_grasps_right,false_grasps \
                = create_correct_false_grasps_mask(grasp, args_train["batch_size"],obj_mask,height,img,VIS=False,mode='unlimited')
            
            grasp_data = {
                'correct': grasp.numpy(),
                'wrong_far_grasps_right': wrong_far_grasps_right.numpy(),
                'wrong_far_grasps_left': wrong_far_grasps_left.numpy(),
                'wrong_mask_grasps_left': wrong_mask_grasps_left.numpy(),
                'wrong_mask_grasps_right': wrong_mask_grasps_right.numpy(),
                'false_grasps': false_grasps.numpy()
            }

            fn = path.split('/')[-1].split('.')[0]
            
            
            np.save(output_dir + fn + '.npy', grasp_data)

def main(args_train):
    device = torch.device(args_train["device"])
    image_transform = get_transform()
    model = BCEGraspTransformer(img_size=args_train["img_size"])
    dataset = JacquardSamples(dataset_root= args_train["split"] ,image_transform=image_transform, num_targets=5, overfit=False,
                              img_size=args_train["img_size"], idx=args_train["num_objects"])
    print(len(dataset))
    device = args_train["device"] if torch.cuda.is_available() else "cpu"
    create_jsons(dataset,args_train, device)

args_train = {
    "split" : "Bottle_train/",
    "device": "cuda",
    "angle_mode": False,
    "img_size": 1120,
    "num_epochs": 100,
    "num_objects": 4,
    "lr": 1e-3,
    "batch_size": 64,
    "print_every_n": 1,
    "experiment_name": "bottle_1_double_sampling"
}


main(args_train)