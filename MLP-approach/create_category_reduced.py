import os 
import json 
import shutil
import cv2




with open('Dataset/simple_jacquard_train.txt', 'r') as file:
    content = file.read()

with open('Dataset/simple_jacquard_test.txt', 'r') as file_test:
    content_test = file_test.read()

# Split the content by lines
lines = content.split('\n')
lines_test = content_test.split('\n')

# Create an empty dictionary
data_dict_train = {'Objects':[]}
data_dict_test = {'Objects':[]}

# Iterate over the lines
for line in lines[1:]:
    # Skip empty lines
    if not line or line == '':
        continue
    data_dict_train['Objects'].append(line)

for line in lines_test[1:]:
    # Skip empty lines
    if not line or line == '':
        continue
    data_dict_test['Objects'].append(line)



base = 'Dataset/categories/'
if os.path.exists(base) == True:
        shutil.rmtree(base)
os.makedirs(base)

full_data_pth = 'Dataset/Full_Dataset/'
AJD_path = os.path.expanduser('~') + '/GraspKpNet/datasets/Jacquard/coco/512_cnt_angle/train/grasps_train2018/'
AJD_path_test= os.path.expanduser('~') + '/GraspKpNet/datasets/Jacquard/coco/512_cnt_angle/test/grasps_test2018/'

splits = ['train', 'test']
#splits = ['train']
for obj in ['Objects']:
        cur_name = obj
        train_obj = data_dict_train[obj]
        test_objs = data_dict_test[obj]
        
        for split in splits : 
                cur_base_dir = base + cur_name + '_' + split + '/'
                if os.path.exists(cur_base_dir) == True:
                        shutil.rmtree(cur_base_dir)
                os.makedirs(cur_base_dir)
                        
        
                if split == 'train': 
                        relevant_objects = train_obj
                else : 
                        relevant_objects = test_objs
                        AJD_path = AJD_path_test
                
                for subset in os.listdir(full_data_pth): 
                        for folder in os.listdir(full_data_pth + subset): 
                                cur_full_name = full_data_pth + subset + '/' + folder
                                for relevant_object in relevant_objects: 
                                        if relevant_object in cur_full_name:
                                                new_dir_name = cur_base_dir + relevant_object + '/'
                                                ajd_pth = AJD_path + '/'.join(cur_full_name.split('/')[2:]) 
                                                try : 
                                                    fs = os.listdir(ajd_pth)
                                                except : 
                                                    print("path not existing", ajd_pth + " skipping therefore") 
                                                    continue
                                                existing_nums = []
                                                for f in fs : 
                                                        if f.endswith('.png'):
                                                                existing_nums.append(int(f.split('_')[0]))      
                                                #import pdb; pdb.set_trace()     
                                                print(new_dir_name) 
                                                try :
                                                    shutil.copytree(cur_full_name, new_dir_name)
                                                except : 
                                                    print('new dir incorrect, skipping', new_dir_name)
                                                    continue
                                                fs = os.listdir(new_dir_name)
                                                for f in fs : 
                                                        start_num = int(f.split('_')[0])        
                                                        if start_num not in existing_nums:
                                                                os.remove(new_dir_name + f)
                                                
                                        
								
                                
		
                
                
                
                
                        
                
