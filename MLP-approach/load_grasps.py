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


def load_jsons(args_train):
    output_dir = 'data/' + args_train['split']
    for f in os.listdir(output_dir):
        fn  = output_dir + f 
        data = np.load(fn,allow_pickle=True)
        
        grasp_correct = data[()]['correct']
        wrong_far_grasps_right = data[()]['wrong_far_grasps_right']
        wrong_far_grasps_left = data[()]['wrong_far_grasps_left']
        wrong_mask_grasps_left = data[()]['wrong_mask_grasps_left']
        wrong_mask_grasps_right = data[()]['wrong_mask_grasps_right']
        false_grasps_combo = data[()]['false_grasps']
        
        breakpoint()
        
    
    
    
load_jsons(args_train) #actualy we just need the split info from args_train
    
    