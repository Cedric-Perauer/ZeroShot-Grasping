import os 
import json 



def create_txt(base_dir,json_file):
        data = json.load(open(json_file))
        data = list(set(data)) 
        deleted_objects = []
        for obj in data  : 
            try : 
                name = obj.split('/')[-2]
            except : 
                pass
            deleted_objects.append(name)
        
        deleted_objects = list(set(deleted_objects))
        
        subfolders = os.listdir(base_dir)   
        all_objects = []
        for subfolder in subfolders:
            if os.path.isdir(base_dir + subfolder):
                objects_ids = os.listdir(base_dir + subfolder)
                all_objects += objects_ids
        
        filtered_objects = []
        
        for obj in all_objects : 
            if obj not in deleted_objects : 
                filtered_objects.append(obj)
        
        txt_name = 'train' if 'train' in json_file else 'test'
        with open('simple_jacquard_'+ txt_name +  '.txt', 'w') as f:
            f.write('Objects\n')
            for fn in filtered_objects:
                f.write(fn + '\n')
                
    
#base_dir = os.path.expanduser('~') + '/GraspKpNet/datasets/Jacquard/coco/512_cnt_angle_orig/train/grasps_train2018/'
base_dir = os.path.expanduser('~') + '/GraspKpNet/datasets/Jacquard/coco/512_cnt_angle_orig/test/grasps_test2018/'
#json_name = 'remove_objects_train.json'
json_name = 'remove_test_images.json'
create_txt(base_dir,json_name)

