import os 
import json 
import shutil
import cv2




with open('categories.txt', 'r') as file:
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

if os.path.exists('new_data') == True:
        shutil.rmtree('new_data')

os.makedirs('new_data')

splits = ['train', 'test']
for obj in data_dict:
        cur_name = obj
        objects = data_dict[obj]
        train_obj = [objects[0]]
        test_objs = objects[1:]
        
        for split in splits : 
                if split == 'train':
                        relevant_objects = train_obj
                else : 
                        relevant_objects = test_objs
                
                new_json_name = 'instances_grasps_train2018_filter.json'
                if split == 'test': 
                        new_json_name = 'instances_grasps_test2018_filter.json'
                
                json_not_filtered = 'instances_grasps_train2018.json'
                if split == 'test': 
                        json_not_filtered = 'instances_grasps_test2018.json'
                        
                pth = 'datasets/Jacquard/coco/512_cnt_angle/train/'
                ##paths for the full dataset
                old_json_name = pth + "instances_grasps_train2018_filter.json"
                img_pths = pth + 'grasps_train2018/'
                old_labels1 = pth + 'train_annotations_0_5'
                old_labels2 = pth + 'train_annotations_6_11'


                labels_0_5_pth = pth + 'labels_0.5/'
                labels_6_11_pth = pth + 'labels_0.5/'


                new_data = {}

                new_data_pth =    'new_data/' + cur_name + '/train/grasps_train2018/'
                new_labels_pth1 = 'new_data/' + cur_name + '/train/train_annotations_0_5/'
                new_labels_pth2 = 'new_data/' + cur_name + '/train/train_annotations_6_11/'
                if split == 'test':
                        new_data_pth = 'new_data/' + cur_name + '/test/grasps_test2018/'
                        new_labels_pth1 =  'new_data/' +  cur_name + '/test/test_annotations_0_5/'
                        new_labels_pth2 =  'new_data/' +  cur_name + '/test/test_annotations_6_11/'
                        

                if os.path.exists(new_data_pth) == True:
                        shutil.rmtree(new_data_pth)
                        
                os.makedirs(new_data_pth)

                if os.path.exists(new_labels_pth1) == True:
                        shutil.rmtree(new_labels_pth1)
                        
                os.makedirs(new_labels_pth1)

                if os.path.exists(new_labels_pth2) == True:
                        shutil.rmtree(new_labels_pth2)
                        
                os.makedirs(new_labels_pth2)


                with open(old_json_name, "r") as f:
                        data = json.load(f)

                new_data['info'] = data['info'].copy()
                new_data['licenses'] = data['licenses'].copy()
                new_data['categories'] = data['categories'].copy()    
                new_data['annotations'] = []
                new_data['images'] = []
                for relevant_object in relevant_objects:
                        img_ids = []
                        for idx in range(len(data['images'])):
                                fn = data['images'][idx]['file_name']
                                if relevant_object in fn : 
                                        new_data['images'].append(data['images'][idx].copy())
                                        img_ids.append(data['images'][idx]['id'])
                                        #new_data['annotations'].append(data['annotations'][idx].copy())
                                        img_name_old = img_pths + data['images'][idx]['file_name']
                                        img = cv2.imread(img_name_old)
                                        img_name_new = new_data_pth + data['images'][idx]['file_name']
                                        first_folder = fn.split('/')[0]	
                                        second_folder = fn.split('/')[1]
                                        num = int(first_folder[-1])
                                        label_folder = new_labels_pth1
                                        label_cat = 'train_annotations_0_5/'
                                        if num > 5:
                                                label_folder = new_labels_pth2
                                                label_cat = 'train_annotations_6_11/'
     
                                        
                                        full_folder_label = label_folder  + first_folder + '/' + second_folder + '/'
                                        full_folder_img = new_data_pth + first_folder + '/' + second_folder + '/'
                                        if os.path.exists(full_folder_img) == False:
                                                os.makedirs(full_folder_img)


                                        source_folder = pth + label_cat + first_folder + '/' + second_folder + '/'
                                        if os.path.exists(full_folder_label) == False:
                                                try : 
                                                        shutil.copytree(source_folder, full_folder_label)
                                                except : 
                                                        import pdb; pdb.set_trace()
                                        cv2.imwrite(img_name_new,img)

                        for j in range(len(data['annotations'])):
                                img_id = data['annotations'][j]['image_id']
                                if img_id in img_ids:
                                        new_data['annotations'].append(data['annotations'][j].copy())
                        

                with open(new_data_pth.split('/')[0] + '/' + cur_name + '/' + split + '/' + new_json_name, "w") as f:
                        json.dump(new_data,f,indent=4)
                
                with open(new_data_pth.split('/')[0] + '/' + cur_name + '/' + split + '/' + json_not_filtered, "w") as f:
                        json.dump(new_data,f,indent=4)
        print(cur_name + "object done")
                
                
                        
                
