import os 
import json 
import shutil
import cv2

new_json_name = 'instances_grasps_train2018_filter.json'
pth = 'datasets/Jacquard/coco/512_cnt_angle/train/'
##paths for the full dataset
old_json_name = pth + "instances_grasps_train2018_filter.json"
img_pths = pth + 'grasps_train2018/'
old_labels1 = pth + 'train_annotations_0_5'
old_labels2 = pth + 'train_annotations_6_11'


labels_0_5_pth = pth + 'labels_0.5/'
labels_6_11_pth = pth + 'labels_0.5/'


new_data = {}

relevant_objects = ['1a2a5a06ce083786581bb5a25b17bed6','1a4daa4904bb4a0949684e7f0bb99f9c','7cc130874942f2132e41f7ed9c5f7eed']

new_data_pth = 'new_data/grasps_train2018/'
new_labels_pth1 = 'new_data/train_annotations_0_5/'
new_labels_pth2 = 'new_data/train_annotations_6_11/'

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
				shutil.copytree(source_folder, full_folder_label)

			cv2.imwrite(img_name_new,img)

	for j in range(len(data['annotations'])):
		img_id = data['annotations'][j]['image_id']
		if img_id in img_ids:
			new_data['annotations'].append(data['annotations'][j].copy())
		

with open(new_data_pth.split('/')[0] + '/' + new_json_name, "w") as f:
        json.dump(new_data,f,indent=4)
	
