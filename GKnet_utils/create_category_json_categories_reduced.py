import os 
import json 
import shutil
import cv2





SIZES = ['Reduced']

with open('simple_jacquard_train.txt', 'r') as file:
    content = file.read()

with open('simple_jacquard_test.txt', 'r') as file_test:
    content_test = file_test.read()

# Split the content by lines
lines = content.split('\n')
lines_test = content_test.split('\n')

# Create an empty dictionary
data_dict = {'Objects':[]}
data_dict_test = {'Objects':[]}

# Iterate over the lines
for line in lines[1:]:
    # Skip empty lines
    if not line or line == '':
        continue
    data_dict['Objects'].append(line)
        

for line_test in lines_test[1:]:
    # Skip empty lines
    if not line_test or line_test == '':
        continue
    data_dict_test['Objects'].append(line_test)

if os.path.exists('new_data') == True:
        shutil.rmtree('new_data')

os.makedirs('new_data')
#splits = ['train', 'test']
splits = ['test']
total_files = 0 
for size in SIZES :
        for obj in ['Objects']:
                cur_name = obj
                train_obj = data_dict[obj]
                test_objs = data_dict_test[obj]
                for split in splits : 
                        print('----------------')
                        print('----------------')
                        print('----------------')
                        print('----------------')
                        print('----------------')
                        print(split)
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
                        if split == 'test':
                                pth = 'datasets/Jacquard/coco/512_cnt_angle/test/'
                                ##paths for the full dataset
                                old_json_name = pth + "instances_grasps_test2018_filter.json"
                                img_pths = pth + 'grasps_test2018/'
                                old_labels1 = pth + 'test_annotations_0_5'
                                old_labels2 = pth + 'test_annotations_6_11'
                        
                        

                        labels_0_5_pth = pth + 'labels_0.5/'
                        labels_6_11_pth = pth + 'labels_0.5/'


                        new_data = {}

                        size_name = str(size) + '_dataset/'
                        new_data_pth =    'new_data/' + size_name + cur_name + '/train/grasps_train2018/'
                        new_labels_pth1 = 'new_data/' + size_name + cur_name + '/train/train_annotations_0_5/'
                        new_labels_pth2 = 'new_data/' + size_name + cur_name + '/train/train_annotations_6_11/'
                        if split == 'test':
                                new_data_pth = 'new_data/'     + size_name  + cur_name + '/test/grasps_test2018/'
                                new_labels_pth1 =  'new_data/' + size_name  +  cur_name + '/test/test_annotations_0_5/'
                                new_labels_pth2 =  'new_data/' + size_name  +  cur_name + '/test/test_annotations_6_11/'
                                

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
                        img_ids = []
                        for idx in range(len(data['images'])):
                                        fn = data['images'][idx]['file_name']
                                        obj_name = fn.split('/')[1]
                                        if obj_name in relevant_objects : 
                                                new_data['images'].append(data['images'][idx].copy())
                                                img_ids.append(data['images'][idx]['id'])
                                                #new_data['annotations'].append(data['annotations'][idx].copy())
                                                img_name_old = img_pths + data['images'][idx]['file_name']
                                                img = cv2.imread(img_name_old)
                                                img_name_new = new_data_pth + data['images'][idx]['file_name']
                                                first_folder = fn.split('/')[0]	
                                                second_folder = fn.split('/')[1]
                                                num = int(first_folder.split('_')[-1])
                                                label_folder = new_labels_pth1
                                                label_cat = 'train_annotations_0_5/'
                                                if split       == 'test':
                                                        label_cat = 'test_annotations_0_5/'
                                                if num > 5:
                                                        label_folder = new_labels_pth2
                                                        label_cat = 'train_annotations_6_11/'
                                                        if split == 'test':
                                                                label_cat = 'test_annotations_6_11/'
        
                                                
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
                                                try : 
                                                        cv2.imwrite(img_name_new,img)
                                                except : 
                                                        breakpoint()
                                                total_files += 1
                                                print("files", total_files)


                        for j in range(len(data['annotations'])):
                                img_id = data['annotations'][j]['image_id']
                                if img_id in img_ids:
                                        new_data['annotations'].append(data['annotations'][j].copy())
                                
                        with open("/".join(new_data_pth.split('/')[:-2]) + '/' + new_json_name, "w") as f:
                                json.dump(new_data,f,indent=4)
                        
                        with open('/'.join(new_data_pth.split('/')[:-2]) + '/' + json_not_filtered, "w") as f:
                                json.dump(new_data,f,indent=4)
                print(cur_name + "object done")

print("total files", total_files)   
                
                        
                
