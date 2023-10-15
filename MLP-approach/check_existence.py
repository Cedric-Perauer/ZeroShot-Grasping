import os 
import json 
import shutil
import cv2

'''
this is a script to check the number of objects that exist within a set 
originally used to debug a mismatch between ajd and jacquard filtered datasets
'''


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

#splits = ['train', 'test']
#splits = ['train']
objects = []

cnt = 0 
splits = ['test']
for obj in ['Objects']:
        cur_name = obj
        train_obj = data_dict_train[obj]
        test_objs = data_dict_test[obj]
        
        for split in splits : 
                cur_base_dir = base + cur_name + '_' + split + '/'

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
                                                objects.append(relevant_object)
                                                ajd_pth = AJD_path + '/'.join(cur_full_name.split('/')[2:]) 
                                                fs = os.listdir(ajd_pth)
                                                try : 
                                                    fs_jac = os.listdir(cur_full_name + '/')
                                                except : 
                                                    breakpoint()
                                                
                                                for f in fs : 
                                                    f = f.replace('RGD',"RGB")
                                                    cnt += 1
                                                    if f not in fs_jac:
                                                        breakpoint()
                                        #else : 
                                        #    breakpoint()
                                                    
print('total test cnt', cnt)
print("num objects", len(objects))
                                        
								
                                
		
                
                
                
                
                        
                
