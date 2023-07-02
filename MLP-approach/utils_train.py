import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import get_transform, get_inv_transform

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
    # Calculate element-wise inequality
    unequal_tensors = torch.all(torch.ne(a[:, None], b), dim=2)

    # Find indices where no tensor in a matches with any tensor in b
    indices_to_keep = torch.all(unequal_tensors, dim=0)

    # Filter tensors in b based on the indices
    filtered_b = b[indices_to_keep]

    return filtered_b

def create_false_points_mask(grasp,mask,bs,img=None,VIS=False):
    inv_transform = get_inv_transform()
    mask = mask.permute(0,2,1).unsqueeze(0)
    if img is not None : 
        img = torch.nn.functional.interpolate(img.unsqueeze(0), (PATCH_DIM, PATCH_DIM), mode="bilinear")[0]
    mask = torch.nn.functional.interpolate(mask, (PATCH_DIM, PATCH_DIM), mode="nearest")[0]
    zero_indices = torch.nonzero(mask[0] != 1) 
    one_indices = torch.nonzero(mask[0] == 1)  
    reshaped_grasps = grasp.reshape(-1,2)

    false_points_object = check_and_remove_tensors(reshaped_grasps, one_indices) ##make sure that no gt grasps are contained due to feature overlap 
    false_points_object = extract_random_elements(false_points_object, bs)
    
    false_points_grasp = check_and_remove_tensors(reshaped_grasps, zero_indices) ##make sure that no gt grasps are contained due to feature overlap 
    false_points_grasp = extract_random_elements(false_points_grasp, bs)
    
    
    ## vis the data 
    if VIS : 
        grasp_vis = torch.zeros((PATCH_DIM, PATCH_DIM))
        for g in grasp:
                g1, g2 = g[0], g[1]
                grasp_vis[g1[0], g1[1]] = 1
                grasp_vis[g2[0], g2[1]] = 1
        grasp_vis = grasp_vis.unsqueeze(0).unsqueeze(0)
        grasp_vis = torch.nn.functional.interpolate(grasp_vis, (PATCH_DIM, PATCH_DIM), mode="nearest").squeeze()
        zeros = torch.zeros(PATCH_DIM, PATCH_DIM, 1)
        grasp_vis = torch.cat([zeros,grasp_vis.cpu().detach().unsqueeze(2), zeros], dim = 2)

        
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
    
    
    return torch.cat([false_points_object,false_points_grasp],dim=0)
            

    
def create_correct_false_points_mask(grasp, bs,mask,img=None,VIS=False):
    false_points_mask = create_false_points_mask(grasp,mask,bs,img,VIS).reshape(-1,2,2)
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