import torch
import numpy as np


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

def create_false_points_mask(grasp,mask,bs):
    zero_indices = torch.nonzero(mask[0] == 0) // 14
    one_indices = torch.nonzero(mask[0] == 1)  // 14
    reshaped_grasps = grasp.reshape(-1,2)
    
    #false_mask = check_and_remove_tensors(one_indices, reshaped_grasps) ##make sure that no gt grasps are contained due to feature overlap 
    false_points_object = check_and_remove_tensors2(reshaped_grasps, one_indices) ##make sure that no gt grasps are contained due to feature overlap 
    false_points_object = extract_random_elements(false_points_object, bs)
    
    false_points_grasp = check_and_remove_tensors2(reshaped_grasps, zero_indices) ##make sure that no gt grasps are contained due to feature overlap 
    false_points_grasp = extract_random_elements(false_points_grasp, bs)
    
    
    return torch.cat([false_points_grasp,false_points_grasp],dim=0)
            

    
def create_correct_false_points_mask(grasp, bs,mask):
    false_points_mask = create_false_points_mask(grasp,mask,bs).reshape(-1,2,2)
    
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