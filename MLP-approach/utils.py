import torch
import numpy as np
import torchvision.transforms.functional as TF
import math
from torchvision import transforms

IMAGE_SIZE = 1120 

def get_transform():
    image_norm_mean = (0.485, 0.456, 0.406)
    image_norm_std = (0.229, 0.224, 0.225)
    image_size = IMAGE_SIZE
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_norm_mean, std=image_norm_std)
    ])
    return image_transform

def get_transform_resized():
    image_norm_mean = (0.485, 0.456, 0.406)
    image_norm_std = (0.229, 0.224, 0.225)
    image_size = 80
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_norm_mean, std=image_norm_std)
    ])
    return image_transform


def get_transform_mask():
    image_size = IMAGE_SIZE
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    return image_transform

def get_transform_resized_mask(image_size):
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    return image_transform


def get_inv_transform():
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.]),
                                   ])
    return invTrans


def distance(ax, ay, bx, by):
    return math.sqrt((by - ay) ** 2 + (bx - ax) ** 2)

def rotated_about(ax, ay, bx, by, angle):
    radius = distance(ax, ay, bx, by)
    angle += math.atan2(ay - by, ax - bx)

    # import pdb; pdb.set_trace()
    try:
        return [
            round(bx.item() + radius * math.cos(angle)),
            round(by.item() + radius * math.sin(angle))
        ]
    except:
        return [
            round(bx + radius * math.cos(angle)),
            round(by + radius * math.sin(angle))
        ]

def create_grasp_rectangle(grasp):
    mids = []
    grasps = []
    centers = []
    gripper_poses = []
    for n, el in enumerate(grasp):
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        x, y, angle, w, h = el
        # if [x,y] in mids: deactivate for the plots
        #    continue
        mids.append([x, y])
        tl = (x - w / 2, y - h / 2)
        bl = (x - w / 2, y + h / 2)
        tr = (x + w / 2, y - h / 2)
        br = (x + w / 2, y + h / 2)
        # use left and right point representation as in GKNet, choose the midpoints of the vertical lines
        mr = ((br[0] + tr[0]) * 0.5, (br[1] + tr[1]) * 0.5)
        ml = ((bl[0] + tl[0]) * 0.5, (bl[1] + tl[1]) * 0.5)

        points = [br, tr, tl, bl]
        grippers = [ml, mr]
        # tl,bl,tr,br = self.rotate_points((x,y),points,angle)
        # print(len(points))
        square_vertices = [rotated_about(pt[0], pt[1], x, y, math.radians(angle)) for pt in points]
        gripper_points = [rotated_about(pt[0], pt[1], x, y, math.radians(angle)) for pt in grippers]
        grasps.append(square_vertices)
        centers.append([x, y, angle, w, h])
        gripper_poses.append(gripper_points)
        # br,tr,tl,bl = square_vertices
    # import pdb; pdb.set_trace()
    #print(grasps[0])
    return grasps, centers, gripper_poses


def get_grasp(grasp_txt, img_size, crop):
    out_grasps = []
    grasping_labels = []
    raw_grasp_labels = []
    gripper_labels = []
    grasp = []
    with open(grasp_txt, 'r') as f:
        lines = f.readlines()
        for l in lines:
            split = l.split(';')
            x, y, angle, w, h = split
            h = h.split('\n')[0]
            x, y, angle, w, h = float(x), float(y), float(angle), float(w), float(h)
            if crop == False:
                grasp.append([x * img_size / 1024., y * img_size / 1024., angle, w * img_size / 1024,
                               h * img_size / 1024])
            else:
                grasp.append([x, y, angle, w, h])

        raw_grasp_labels.append(grasp)
        if crop == False:
            grasps, raw_labels, gripper_points = create_grasp_rectangle(grasp)
            grasping_labels.append(grasps)
            gripper_labels.append(gripper_points)
            out_grasps.append(grasp)
        return np.array(gripper_labels), np.array(out_grasps), np.array(grasping_labels)

def augment_image(image, angle=10):
    ##rotate the image
    image_augmented = TF.rotate(image, angle)
    return image_augmented

def get_augmented_angles(augmented_gknet_label, angle, img_size):
    gknet_label = augmented_gknet_label.copy()[:4]
    augmented_angle = augmented_gknet_label[2] - angle  # just change the angle of the label here
    augmented_angle2 = augmented_angle
    if augmented_angle < 0:
        augmented_angle2 = augmented_angle + 180
    else:
        augmented_angle2 = augmented_angle - 180

    xr, yr = rotated_about(augmented_gknet_label[0], augmented_gknet_label[1], img_size / 2.,
                                img_size / 2., math.radians(-angle))
    x = xr / img_size
    y = yr / img_size
    angle_cos = math.cos(math.radians(augmented_angle))
    angle_sin = math.sin(math.radians(augmented_angle))

    angle_cos2 = math.cos(math.radians(augmented_angle2))
    angle_sin2 = math.sin(math.radians(augmented_angle2))
    wr = augmented_gknet_label[3]
    w = augmented_gknet_label[3] / img_size
    augmented_gknet_label = torch.tensor([x, y, angle_cos, angle_sin, w])
    augmented_gknet_label2 = torch.tensor([x, y, angle_cos2, angle_sin2, w])

    xt, yt, angle_label, wt = gknet_label
    gknet_label = [xt / img_size, yt / img_size, math.cos(math.radians(angle_label)),
                   math.sin(math.radians(angle_label)), wt / img_size]
    '''
    if self.crop == False :
        grasps_ref = grasps_ref * self.img_size/1024
    '''
    lxt, lyt = xr - wr / 2., yr
    rxt, ryt = xr + wr / 2., yr
    rxt, ryt = rotated_about(rxt, ryt, float(xr), float(yr), math.radians(augmented_angle))
    lxt, lyt = rotated_about(lxt, lyt, float(xr), float(yr), math.radians(augmented_angle))
    return augmented_gknet_label, augmented_gknet_label2, gknet_label,\
           torch.tensor([[lxt, lyt], [rxt, ryt]]), torch.tensor([[rxt, ryt], [lxt, lyt]])


