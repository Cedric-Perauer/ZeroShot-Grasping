import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import get_transform, get_inv_transform
import time 
from shapely.geometry import Polygon
from metrics_utils import oriented_bounding_box_iou, grasp_correct_full

IMAGE_SIZE = 1120
PATCH_DIM = IMAGE_SIZE  // 14

def extract_random_elements(tensor, num_elements):
    num_total_elements = tensor.shape[0]  # Total number of elements in the tensor
    indices = torch.randperm(num_total_elements)[:num_elements]  # Generate random indices
    return tensor[indices,:]


def create_false_points(minx, maxx, miny, maxy, bs):
    x_false = torch.randint(minx, maxx, (bs * 2, 1))
    y_false = torch.randint(miny, maxy, (bs * 2, 1))
    return torch.cat([x_false, y_false], dim=1).reshape(bs, 2, 2)

def check_and_remove_tensors(a, b):
    for tensor_a in a:
        for i, tensor_b in enumerate(b):
            if torch.all(torch.eq(tensor_a, tensor_b)):
                b = torch.cat((b[:i], b[i+1:]), dim=0)
                break

    return b

def check_and_remove_tensors2(a, b):
    mask = ~torch.all(torch.eq(b.unsqueeze(1), a), dim=-1).any(dim=1)
    b = b[mask]
    return b

def create_false_points_mask(grasp,mask,bs,img=None,VIS=False):
    inv_transform = get_inv_transform()
    mask = mask.permute(0,2,1).unsqueeze(0)
    
    
    #get left and right grasp points for 1 point method 
    grasps_left = torch.empty((grasp.shape[0],2))
    grasps_right = torch.empty((grasp.shape[0],2))  
    for i,g in enumerate(grasp):
                g1, g2 = g[0], g[1]
                if g1[1] < g2[1]:
                    grasps_left[i] = g1
                    grasps_right[i] = g2
                elif g1[1] > g2[1]:
                    grasps_left[i] = g2
                    grasps_right[i] = g1
                else : 
                    #check for y 
                    if g1[0] >= g2[0]:
                        grasps_left[i] = g1
                        grasps_right[i] = g2
                    else :
                        grasps_left[i] = g2
                        grasps_right[i] = g1
    
    
    if img is not None : 
        img = torch.nn.functional.interpolate(img.unsqueeze(0), (PATCH_DIM, PATCH_DIM), mode="bilinear")[0]
    mask = torch.nn.functional.interpolate(mask, (PATCH_DIM, PATCH_DIM), mode="nearest")[0]
    zero_indices = torch.nonzero(mask[0] != 1) 
    one_indices = torch.nonzero(mask[0] == 1)  
    reshaped_grasps = grasp.reshape(-1,2)

    #start = time.time()
    false_points_object = check_and_remove_tensors2(reshaped_grasps, one_indices) ##make sure that no gt grasps are contained due to feature overlap
    false_points_object = extract_random_elements(false_points_object, bs)
    num = false_points_object.shape[0]
    
    second_pts = bs 
    if num != bs :
        second_pts = second_pts + (bs - num)
        
    false_points_grasp = check_and_remove_tensors2(reshaped_grasps, zero_indices) ##make sure that no gt grasps are contained due to feature overlap
    false_points_grasp = extract_random_elements(false_points_grasp, second_pts)
    #end = time.time() - start 
    #print("sampling took ", end * 1000 ," ms")
    
    ## vis the data 
    if VIS : 
        grasp_vis_left = torch.zeros((PATCH_DIM, PATCH_DIM))
        grasp_vis_right = torch.zeros((PATCH_DIM, PATCH_DIM))
        for g in grasps_left :
            grasp_vis_left[int(g[0]),int(g[1])] = 1
        
        for g in grasps_right :
            grasp_vis_right[int(g[0]),int(g[1])] = 1
        
                        
        grasp_vis_left = grasp_vis_left.unsqueeze(0).unsqueeze(0)
        grasp_vis_right = grasp_vis_right.unsqueeze(0).unsqueeze(0)
        grasp_vis_left = torch.nn.functional.interpolate(grasp_vis_left, (PATCH_DIM, PATCH_DIM), mode="nearest").squeeze()
        grasp_vis_right = torch.nn.functional.interpolate(grasp_vis_right, (PATCH_DIM, PATCH_DIM), mode="nearest").squeeze()
        zeros = torch.zeros(PATCH_DIM, PATCH_DIM, 1)
        grasp_vis = torch.cat([grasp_vis_right.cpu().detach().unsqueeze(2),grasp_vis_left.cpu().detach().unsqueeze(2), grasp_vis_right.cpu().detach().unsqueeze(2)], dim = 2)

        
        one_indices_vis = torch.nonzero(mask[0] == 1)  
        mask_tensor = mask[0] == 1     
        mask_tensor = torch.logical_not(mask_tensor)
        zero_indices_vis = torch.nonzero(mask_tensor)

        

        mask_vis = torch.zeros((PATCH_DIM, PATCH_DIM))
        for idcs in one_indices_vis:
            mask_vis[idcs[0], idcs[1]] = 1
        mask_vis1 = mask_vis.unsqueeze(0).unsqueeze(0)
        mask_vis = torch.nn.functional.interpolate(mask_vis1, (IMAGE_SIZE, IMAGE_SIZE), mode="bilinear").squeeze()
        zeros = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, 1)
        mask_vis = torch.cat([mask_vis.cpu().detach().unsqueeze(2), zeros, zeros], dim = 2)
        mask_vis1 = torch.cat([mask_vis1.squeeze().cpu().detach().unsqueeze(2), torch.zeros(PATCH_DIM,PATCH_DIM,1), torch.zeros(PATCH_DIM,PATCH_DIM,1)], dim = 2)
        
        
        false_objects_vis = torch.zeros((PATCH_DIM, PATCH_DIM))
        for idcs in false_points_object:
            false_objects_vis[idcs[0], idcs[1]] = 1
        false_objects_vis = false_objects_vis.unsqueeze(0).unsqueeze(0).squeeze()
        #false_objects_vis = torch.nn.functional.interpolate(false_objects_vis, (PATCH_DIM, PATCH_DIM), mode="bilinear").squeeze()
        zeros = torch.zeros(PATCH_DIM, PATCH_DIM, 1)
        false_objects_vis = torch.cat([zeros, zeros,false_objects_vis.cpu().detach().unsqueeze(2)], dim = 2)
        
        false_points_vis = torch.zeros((PATCH_DIM, PATCH_DIM))
        for idcs in false_points_grasp:
            false_points_vis[idcs[0], idcs[1]] = 1
        false_points_vis = false_points_vis.unsqueeze(0).unsqueeze(0).squeeze()
        #false_points_vis = torch.nn.functional.interpolate(false_points_vis, (IMAGE_SIZE, IMAGE_SIZE), mode="nearest-exact").squeeze()
        zeros = torch.zeros(PATCH_DIM, PATCH_DIM, 1)
        #ones = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, 1)
        false_points_vis = torch.cat([zeros, zeros,false_points_vis.cpu().detach().unsqueeze(2)], dim = 2)
        
        #show_img = org_image + 0.7*preds.cpu().detach().numpy() 
        img =torch.permute(inv_transform(img), (1, 2, 0)).cpu().numpy()
        show_img = 0.7 * img + 0.3 * grasp_vis.numpy()
        #show_img = 0.7 * img + 0.2 * grasp_vis.numpy() + 0.2 * mask_vis1.numpy() + 0.5 * false_points_vis.numpy() 
        show_img2 = 0.7 * img + 0.2 * grasp_vis.numpy() + 0.2 * mask_vis1.numpy() + 0.5 * false_objects_vis.numpy() 
        #show_img2 = mask_vis.numpy()
        #show_img = mask_vis1.numpy()
        #show_img = org_image + 0.7*origin_point2 + 0.7*origin_point
        print("Num false points on object and grasp :",false_points_object.shape[0],false_points_grasp.shape[0])
        fig, axs = plt.subplots(1,2)
        fig.suptitle('Vertically stacked subplots')
        axs[0].imshow(show_img)
        axs[1].imshow(show_img2)
        plt.show()
    
    
    return torch.cat([false_points_object,false_points_grasp],dim=0), grasps_left, grasps_right
            

def create_false_grasps_mask(grasp,mask,bs,height,img=None,VIS=False,img_size=1120):
    inv_transform = get_inv_transform()
    mask = mask.permute(0,2,1).unsqueeze(0)
    
    
    #get left and right grasp points for 1 point method 
    grasps_left = torch.empty((grasp.shape[0],2))
    grasps_right = torch.empty((grasp.shape[0],2))  
    for i,g in enumerate(grasp):
                g1, g2 = g[0], g[1]
                if g1[1] < g2[1]:
                    grasps_left[i] = g1
                    grasps_right[i] = g2
                elif g1[1] > g2[1]:
                    grasps_left[i] = g2
                    grasps_right[i] = g1
                else : 
                    #check for y 
                    if g1[0] >= g2[0]:
                        grasps_left[i] = g1
                        grasps_right[i] = g2
                    else :
                        grasps_left[i] = g2
                        grasps_right[i] = g1
    
    
    if img is not None : 
        img = torch.nn.functional.interpolate(img.unsqueeze(0), (PATCH_DIM, PATCH_DIM), mode="bilinear")[0]
    mask = torch.nn.functional.interpolate(mask, (PATCH_DIM, PATCH_DIM), mode="nearest")[0]
    zero_indices = torch.nonzero(mask[0] != 1) 
    one_indices = torch.nonzero(mask[0] == 1)  
    reshaped_grasps = grasp.reshape(-1,2)

    #start = time.time()
    false_points_object = check_and_remove_tensors2(reshaped_grasps, one_indices) ##make sure that no gt grasps are contained due to feature overlap
    false_points_object = extract_random_elements(false_points_object, bs)
    num = false_points_object.shape[0]
    
    second_pts = bs 
    if num != bs :
        second_pts = second_pts + (bs - num)
        
    false_points_grasp = check_and_remove_tensors2(reshaped_grasps, zero_indices) ##make sure that no gt grasps are contained due to feature overlap
    false_points_grasp = extract_random_elements(false_points_grasp, second_pts)
    
    ##randomly arrange tensor 
    # Get the number of rows (64 in this case)
    num_rows = grasps_left.size(0)

    # Generate a random permutation of indices
    random_indices = torch.randperm(num_rows)

    # Use the permutation to shuffle the rows of the tensor
    grasps_left = grasps_left[random_indices]
    random_indices = torch.randperm(num_rows)
    grasps_right = grasps_right[random_indices]   
    
    
    ##combine left and right with wrong points on mask 
    min_dim = min(grasps_left.shape[0],false_points_object.shape[0])
    wrong_mask_grasps_right = torch.cat((grasps_right[:min_dim].unsqueeze(2), false_points_object[:min_dim].unsqueeze(2)), dim=2)
    wrong_mask_grasps_left = torch.cat((grasps_left[:min_dim].unsqueeze(2), false_points_object[:min_dim].unsqueeze(2)), dim=2)
    
    #combine left and right with wrong points far away from mask and non grasp 
    wrong_far_grasps_right = torch.cat((grasps_right[:min_dim].unsqueeze(2), false_points_grasp[:min_dim].unsqueeze(2)), dim=2)
    wrong_far_grasps_left = torch.cat((grasps_left[:min_dim].unsqueeze(2), false_points_grasp[:min_dim].unsqueeze(2)), dim=2)

    #combine left and right points in different ways to form incorrect grasp pairs, use metrics to check for true incorrect grasp pairs
    wrong_left_right_grasp = torch.cat((grasps_left.unsqueeze(2), grasps_right.unsqueeze(2)), dim=2)
    wrong_right_left_grasp = torch.cat((grasps_right.unsqueeze(2), grasps_left.unsqueeze(2)), dim=2)
    
    #go through each pair and check if they are invalid grasps, otherwise we need to discard them 
    height_average = sum(height)/len(height) * 2
    false_grasps = []
    for wrong_grasp in wrong_left_right_grasp :
        correct_flag = False
        for gt in grasp :
            corner_points, corner_points_pred, correct, iou, angle_diff = grasp_correct_full(wrong_grasp[0] * 1120/80. + 7, wrong_grasp[1] * 1120/80. + 7, 
                                                                    gt * 1120/80. + 7,height_average /2. * img_size ) 
            if correct :
                correct_flag = True
                break 
        
        if correct_flag == False :
            ##add to false grasps 
            false_grasps.append(wrong_grasp)
    
    for wrong_grasp in wrong_right_left_grasp :
        correct_flag = False
        for gt in grasp :
            corner_points, corner_points_pred, correct, iou, angle_diff = grasp_correct_full(wrong_grasp[0] * 1120/80. + 7, wrong_grasp[1] * 1120/80. + 7, 
                                                                    gt * 1120/80. + 7,height_average /2. * img_size ) 
            if correct :
                correct_flag = True
                break 
        
        if correct_flag == False :
            ##add to false grasps 
            false_grasps.append(wrong_grasp)
    
    false_grasps_tensor = torch.zeros((len(false_grasps), 2, 2))
    for idx,wrong_grasp in enumerate(false_grasps) :
        false_grasps_tensor[idx] = wrong_grasp 
    
    ##concatenate points together with total size of batch size
    bs_6 = int(bs / 6) 
    wrong_far_grasps = torch.cat((wrong_far_grasps_right[:bs_6],wrong_far_grasps_left[:bs_6]), dim=0)
    wrong_mask_grasps = torch.cat((wrong_mask_grasps_right[:bs_6],wrong_mask_grasps_left[:bs_6]), dim=0)
    remaining = bs - (bs_6 * 4) #size of false grasps tensor to combine correct shape 
    false_grasps_tensor = false_grasps_tensor[:remaining]
    false_grasps_total = torch.cat([wrong_far_grasps, wrong_mask_grasps, false_grasps_tensor], dim=0)
    
    ## vis the data 
    if VIS : 
        grasp_vis_left = torch.zeros((PATCH_DIM, PATCH_DIM))
        grasp_vis_right = torch.zeros((PATCH_DIM, PATCH_DIM))
        for g in grasps_left :
            grasp_vis_left[int(g[0]),int(g[1])] = 1
        
        for g in grasps_right :
            grasp_vis_right[int(g[0]),int(g[1])] = 1
        
                        
        grasp_vis_left = grasp_vis_left.unsqueeze(0).unsqueeze(0)
        grasp_vis_right = grasp_vis_right.unsqueeze(0).unsqueeze(0)
        grasp_vis_left = torch.nn.functional.interpolate(grasp_vis_left, (PATCH_DIM, PATCH_DIM), mode="nearest").squeeze()
        grasp_vis_right = torch.nn.functional.interpolate(grasp_vis_right, (PATCH_DIM, PATCH_DIM), mode="nearest").squeeze()
        zeros = torch.zeros(PATCH_DIM, PATCH_DIM, 1)
        grasp_vis = torch.cat([grasp_vis_right.cpu().detach().unsqueeze(2),grasp_vis_left.cpu().detach().unsqueeze(2), grasp_vis_right.cpu().detach().unsqueeze(2)], dim = 2)

        
        one_indices_vis = torch.nonzero(mask[0] == 1)  
        mask_tensor = mask[0] == 1     
        mask_tensor = torch.logical_not(mask_tensor)
        zero_indices_vis = torch.nonzero(mask_tensor)

        

        mask_vis = torch.zeros((PATCH_DIM, PATCH_DIM))
        for idcs in one_indices_vis:
            mask_vis[idcs[0], idcs[1]] = 1
        mask_vis1 = mask_vis.unsqueeze(0).unsqueeze(0)
        mask_vis = torch.nn.functional.interpolate(mask_vis1, (IMAGE_SIZE, IMAGE_SIZE), mode="bilinear").squeeze()
        zeros = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, 1)
        mask_vis = torch.cat([mask_vis.cpu().detach().unsqueeze(2), zeros, zeros], dim = 2)
        mask_vis1 = torch.cat([mask_vis1.squeeze().cpu().detach().unsqueeze(2), torch.zeros(PATCH_DIM,PATCH_DIM,1), torch.zeros(PATCH_DIM,PATCH_DIM,1)], dim = 2)
        
        
        false_objects_vis = torch.zeros((PATCH_DIM, PATCH_DIM))
        for idcs in false_points_object:
            false_objects_vis[idcs[0], idcs[1]] = 1
        false_objects_vis = false_objects_vis.unsqueeze(0).unsqueeze(0).squeeze()
        #false_objects_vis = torch.nn.functional.interpolate(false_objects_vis, (PATCH_DIM, PATCH_DIM), mode="bilinear").squeeze()
        zeros = torch.zeros(PATCH_DIM, PATCH_DIM, 1)
        false_objects_vis = torch.cat([zeros, zeros,false_objects_vis.cpu().detach().unsqueeze(2)], dim = 2)
        
        false_points_vis = torch.zeros((PATCH_DIM, PATCH_DIM))
        for idcs in false_points_grasp:
            false_points_vis[idcs[0], idcs[1]] = 1
        false_points_vis = false_points_vis.unsqueeze(0).unsqueeze(0).squeeze()
        #false_points_vis = torch.nn.functional.interpolate(false_points_vis, (IMAGE_SIZE, IMAGE_SIZE), mode="nearest-exact").squeeze()
        zeros = torch.zeros(PATCH_DIM, PATCH_DIM, 1)
        #ones = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, 1)
        false_points_vis = torch.cat([zeros, zeros,false_points_vis.cpu().detach().unsqueeze(2)], dim = 2)
        
        #show_img = org_image + 0.7*preds.cpu().detach().numpy() 
        img =torch.permute(inv_transform(img), (1, 2, 0)).cpu().numpy()
        #show_img = 0.7 * img + 0.3 * grasp_vis.numpy()
        show_img = 0.7 * img + 0.2 * grasp_vis.numpy() + 0.2 * mask_vis1.numpy() + 0.5 * false_points_vis.numpy() 
        show_img2 = 0.7 * img + 0.2 * grasp_vis.numpy() + 0.2 * mask_vis1.numpy() + 0.5 * false_objects_vis.numpy() 
        #show_img2 = mask_vis.numpy()
        #show_img = mask_vis1.numpy()
        #show_img = org_image + 0.7*origin_point2 + 0.7*origin_point
        print("Num false points on object and grasp :",false_points_object.shape[0],false_points_grasp.shape[0])
        fig, axs = plt.subplots(1,2)
        fig.suptitle('Vertically stacked subplots')
        axs[0].imshow(show_img)
        axs[1].imshow(show_img2)
        plt.show()
    
    
    return false_grasps_total, grasps_left, grasps_right

    
def create_correct_false_points_mask(grasp, bs,mask,img=None,VIS=False):
    false_points_mask, grasps_left, grasps_right = create_false_points_mask(grasp,mask,bs,img,VIS)
    try : 
        false_points_mask = false_points_mask.reshape(-1,2,2)
    except : 
        import pdb; pdb.set_trace()
    
    return false_points_mask, grasps_left, grasps_right
    
def create_correct_false_grasps_mask(grasp, bs,mask,height,img=None,VIS=False):
    false_points_mask, _, _  = create_false_grasps_mask(grasp,mask,bs,height,img,VIS)

    
    return false_points_mask


def create_correct_false_points(grasp, bs):
    false_points = create_false_points(grasp[:, :, 0].min().item(), grasp[:, :, 0].max().item(),
                                       grasp[:, :, 1].min().item(), grasp[:, :, 1].max().item(),
                                       bs)
    i = 0
    while i < grasp.shape[0]:
        i_diff = (np.abs(false_points - grasp[i, :, :]).reshape(bs, 4).sum(1) == 0).sum() ## boolean mask 
        if i_diff > 0: ##if one of the false points is equal to grasp point, select points again 
            false_points = create_false_points(grasp[:, :, 0].min().item(), grasp[:, :, 0].max().item(),
                                               grasp[:, :, 1].min().item(), grasp[:, :, 1].max().item(),
                                               bs)
            
            i = 0
            print("tekrarladim")
        else:
            i = i + 1
    return false_points