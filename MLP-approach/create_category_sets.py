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

if os.path.exists('Dataset/categories') == True:
        shutil.rmtree('Dataset/categories')

os.makedirs('new_data')
base = 'Dataset/categories/'
full_data_pth = 'Dataset/Full_Dataset/'

splits = ['train', 'test']
for obj in data_dict:
        cur_name = obj
        objects = data_dict[obj]
        train_obj = [objects[0]]
        test_objs = objects[1:]
        
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
                                                shutil.copytree(cur_full_name, new_dir_name)
                                        
								
                                
		
                
                
                
                
                        
                
