import os 
import cv2  


jacquard_root = os.path.expanduser("~") + '/GraspKpNet/datasets/Jacquard/coco/512_cnt_angle_orig/train/grasps_train2018/'
subfolders = os.listdir(jacquard_root)
subfolders = [folder for folder in subfolders if os.path.isdir(jacquard_root + folder)]

for subfolder in subfolders:
    objects = os.listdir(jacquard_root + subfolder)
    for obj in objects:
        if os.path.isdir(jacquard_root + subfolder + '/' + obj) : 
                files = os.listdir(jacquard_root + subfolder + '/' + obj)
                for file in files[:1]:
                        fn = jacquard_root + subfolder + '/' + obj + '/' + file 
                        print(" ---- current object id --- ")
                        print(obj)
                        img = cv2.imread(fn)
                        cv2.imshow(obj, img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
