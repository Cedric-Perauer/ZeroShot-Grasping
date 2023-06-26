import torch
import numpy as np


def create_false_points(minx, maxx, miny, maxy, bs):
    x_false = torch.randint(minx, maxx, (bs * 2, 1))
    y_false = torch.randint(miny, maxy, (bs * 2, 1))
    return torch.cat([x_false, y_false], dim=1).reshape(bs, 2, 2)

def create_correct_false_points(grasp, bs):
    false_points = create_false_points(grasp[:, :, 0].min().item(), grasp[:, :, 0].max().item(),
                                       grasp[:, :, 1].min().item(), grasp[:, :, 1].max().item(),
                                       bs)
    i = 0
    while i < grasp.shape[0]:
        i_diff = (np.abs(false_points - grasp[i, :, :]).reshape(bs, 4).sum(1) == 0).sum()
        if i_diff > 0:
            false_points = create_false_points(grasp[:, :, 0].min().item(), grasp[:, :, 0].max().item(),
                                               grasp[:, :, 1].min().item(), grasp[:, :, 1].max().item(),
                                               bs)
            i = 0
            print("tekrarladim")
        else:
            i = i + 1
    return false_points