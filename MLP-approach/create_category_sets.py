import os 
import json 
import shutil
import cv2




with open('Dataset/categories.txt', 'r') as file:
    content = file.read()

# Split the content by lines
lines = content.split('\n')

# Create an empty dictionary
data_dict = {}

# Iterate over the lines
for line in lines:
    # Skip empty lines
    if not line:
        continue

    # Check if the line is a name
    if line[0].isalpha():
        current_name = line
        data_dict[current_name] = []
    else:
        # Add the object ID to the current name's list
        data_dict[current_name].append(line)


base = 'Dataset/categories/'
if os.path.exists(base) == True:
        shutil.rmtree(base)
os.makedirs(base)

full_data_pth = 'Dataset/Full_Dataset/'
AJD_path = os.path.expanduser('~') + '/GraspKpNet/datasets/Jacquard/coco/512_cnt_angle/train/grasps_train2018/'

splits = ['train', 'test']
for obj in data_dict:
        cur_name = obj
        objects = data_dict[obj]
        train_obj = objects[:4]
        test_objs = objects[4:]
        
        for split in splits : 
                cur_base_dir = base + cur_name + '_' + split + '/'
                if os.path.exists(cur_base_dir) == True:
                        shutil.rmtree(cur_base_dir)
                os.makedirs(cur_base_dir)
                        
        
                if split == 'train': 
                        relevant_objects = train_obj
                else : 
                        relevant_objects = test_objs
                
                for subset in os.listdir(full_data_pth): 
                        for folder in os.listdir(full_data_pth + subset): 
                                cur_full_name = full_data_pth + subset + '/' + folder
                                for relevant_object in relevant_objects: 
                                        if relevant_object in cur_full_name:
                                                new_dir_name = cur_base_dir + relevant_object + '/'
                                                ajd_pth = AJD_path + '/'.join(cur_full_name.split('/')[2:]) 
                                                fs = os.listdir(ajd_pth)
                                                existing_nums = []
                                                for f in fs : 
                                                        if f.endswith('.png'):
                                                                existing_nums.append(int(f.split('_')[0]))      
                                                #import pdb; pdb.set_trace()     
                                                shutil.copytree(cur_full_name, new_dir_name)
                                                fs = os.listdir(new_dir_name)
                                                for f in fs : 
                                                        start_num = int(f.split('_')[0])        
                                                        if start_num not in existing_nums:
                                                                os.remove(new_dir_name + f)
                                                
                                        
								
                                
		
                
                
                
                
                        
                
