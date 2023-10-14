import torch
import numpy as np 
import matplotlib.pyplot as plt
from utils import *
import time
from metrics_utils import *

def get_features(data, model, device, args_infer):
    img = data["img"].to(device)
    img = torch.permute(img, (0, 2, 1))
    grasp = data["points_grasp"] // 14
    mask = data["mask"]
    height = data['height']
    corners = data['corners']
    grasp_inv = torch.cat([grasp[:, 1, :].unsqueeze(1), grasp[:, 0, :].unsqueeze(1)], dim=1)
    grasp = torch.cat([grasp, grasp_inv], dim=0)
    #features, clk = model.forward_dino_features(img.unsqueeze(0))

    #features = features.squeeze().reshape(args_infer["img_size"] // 14, args_infer["img_size"] // 14, 768)

    return img, mask, grasp, height, corners

def get_unet_preds(unet,valid_pts_pred,mask,img,args_infer):
    input_masks = torch.zeros((valid_pts_pred.shape[0],2,80,80))
    for i in range(valid_pts_pred.shape[0]):
        input_mask = torch.zeros(1,2,80,80).to(mask.device)
        input_mask[0] = mask
        x,y = valid_pts_pred[i,0,1], valid_pts_pred[i,0,0   ]
        input_mask[0,1,int(x.cpu().item()),int(y.cpu().item())] = 1 
        input_masks[i] = input_mask
    model_input = input_masks.to(mask.device)
    if args_infer['rgb']:
        img = img.unsqueeze(0)
        img = img.repeat(model_input.shape[0],1,1,1)
        model_input = torch.cat([model_input,img], dim=1)
    
    preds = unet(model_input)
    max_indices_1d = torch.argmax(preds.view(valid_pts_pred.shape[0], -1), dim=1)
    y_indices = max_indices_1d // 80
    x_indices = max_indices_1d % 80
    conf_vals = preds[torch.arange(preds.shape[0]), :, x_indices, y_indices] 
    max_indices = torch.stack((x_indices, y_indices), dim=-1), conf_vals
    
    return max_indices


def test_single_point(grasp, mask, device, features, model, args_infer, point_idx,heights,single_point=None):
    if single_point is not None : 
        pass
    else : 
        single_point = grasp[point_idx][0]
    single_point_gt = grasp[point_idx][1]
    height = heights[point_idx]
    single_point = single_point.unsqueeze(0).repeat(args_infer["img_size"] // 14 * args_infer["img_size"] // 14,
                                                    1).unsqueeze(1)
    single_point_gt = single_point_gt.unsqueeze(0).repeat(args_infer["img_size"] // 14 * args_infer["img_size"] // 14,
                                                          1).unsqueeze(1)
    all_points = torch.tensor(
        [[i, j] for j in range(args_infer["img_size"] // 14) for i in range(args_infer["img_size"] // 14)]).unsqueeze(1)
    
    all_points = torch.cat([single_point, all_points], dim=1).to(device)
    mean_feats = []
    dif = (all_points[:, 0, :] - all_points[:, 1, :]).type(torch.float32).norm(p=2, dim=1)
    dif_n = (dif / mask).unsqueeze(1)
    for i in range(all_points.shape[0]):
        imix = all_points[i, :, 0].min()
        imax = all_points[i, :, 0].max()
        ymix = all_points[i, :, 1].min()
        ymax = all_points[i, :, 1].max()
        features_i = features[imix:imax + 1, ymix:ymax + 1, :]
        # attn_i = attn_norms[imix:imax+1, ymix:ymax+1].mean()
        features_i = features_i.reshape(features_i.shape[0] * features_i.shape[1], features_i.shape[2]).mean(0)
        if i == 0:
            mean_feats = features_i.unsqueeze(0)
        else:
            mean_feats = torch.cat([mean_feats, features_i.unsqueeze(0)], dim=0)
    with torch.no_grad():
        preds = model(mean_feats.to(device), dif_n.to(device))
    return preds.squeeze(), dif_n.squeeze(), single_point, single_point_gt, height

def test_two_points(grasp, mask, device, features, model, args_infer, point_idx,single_point=None):
    if single_point is not None : 
        pass
    else : 
        single_point = grasp[point_idx]
    
    single_point = single_point.unsqueeze(0).repeat(args_infer["img_size"] // 14 * args_infer["img_size"] // 14,
                                                    1).unsqueeze(1)
    all_points = torch.tensor(
        [[i, j] for j in range(args_infer["img_size"] // 14) for i in range(args_infer["img_size"] // 14)]).unsqueeze(1)
    
    all_points = torch.cat([single_point, all_points], dim=1).to(device)
    mean_feats = []
    dif = (all_points[:, 0, :] - all_points[:, 1, :]).type(torch.float32).norm(p=2, dim=1)
    dif_n = (dif / mask).unsqueeze(1)
    for i in range(all_points.shape[0]):
        imix = int(all_points[i, :, 0].min().cpu())
        imax = int(all_points[i, :, 0].max().cpu())
        ymix = int(all_points[i, :, 1].min().cpu())
        ymax = int(all_points[i, :, 1].max().cpu())
        try : 
            features_i = features[imix:imax + 1, ymix:ymax + 1, :]
        except : 
            import pdb; pdb.set_trace()
        # attn_i = attn_norms[imix:imax+1, ymix:ymax+1].mean()
        features_i = features_i.reshape(features_i.shape[0] * features_i.shape[1], features_i.shape[2]).mean(0)
        if i == 0:
            mean_feats = features_i.unsqueeze(0)
        else:
            mean_feats = torch.cat([mean_feats, features_i.unsqueeze(0)], dim=0)
    with torch.no_grad():
        preds = model(mean_feats.to(device), dif_n.to(device))
    return preds.squeeze(), dif_n.squeeze(), single_point, None

def test_single_point_constraint(grasp, mask, device, features, model, args_infer, point_idx,constrain_mode=True):
    single_point = grasp[point_idx].reshape(2,)
    
    
    if constrain_mode : 
        all_points = grasp
        single_point = single_point.unsqueeze(0).repeat(all_points.shape[0],
                                                    1).unsqueeze(1)
        
        
    else : 
        all_points = torch.tensor(
        [[i, j] for j in range(args_infer["img_size"] // 14) for i in range(args_infer["img_size"] // 14)]).unsqueeze(1)
        single_point = single_point.unsqueeze(0).repeat(args_infer["img_size"] // 14 * args_infer["img_size"] // 14,
                                                    1).unsqueeze(1)
        
    all_points = torch.cat([single_point, all_points], dim=1).to(device)
    mean_feats = []
    dif = (all_points[:, 0, :] - all_points[:, 1, :]).type(torch.float32).norm(p=2, dim=1)
    dif_n = (dif / mask).unsqueeze(1)
    
    imins = []
    imaxs = []
    start = time.time()
    for i in range(all_points.shape[0]):
        imix = int(all_points[i, :, 0].min().cpu())
        imax = int(all_points[i, :, 0].max().cpu())
        ymix = int(all_points[i, :, 1].min().cpu())
        ymax = int(all_points[i, :, 1].max().cpu())
        features_i = features[imix:imax + 1, ymix:ymax + 1, :]
        imins.append(imix)
        imaxs.append(imax)
        # attn_i = attn_norms[imix:imax+1, ymix:ymax+1].mean()
        features_i = features_i.reshape(features_i.shape[0] * features_i.shape[1], features_i.shape[2]).mean(0)
        if i == 0:
            mean_feats = features_i.unsqueeze(0)
        else:
            mean_feats = torch.cat([mean_feats, features_i.unsqueeze(0)], dim=0)
    end = time.time() - start
    


    with torch.no_grad():
        preds = model(mean_feats.to(device), dif_n.to(device))
    return preds.squeeze(), dif_n.squeeze(), single_point, None


def test_single_point_constraint_left_right(grasp_left,grasp_right, mask, device, features, model, args_infer, point_idx,constrain_mode=True):
    single_point = grasp_left[point_idx].reshape(2,)
    
    
    if constrain_mode : 
        all_points = grasp_right
        single_point = single_point.unsqueeze(0).repeat(all_points.shape[0],
                                                    1).unsqueeze(1)
        
        
    else : 
        all_points = torch.tensor(
        [[i, j] for j in range(args_infer["img_size"] // 14) for i in range(args_infer["img_size"] // 14)]).unsqueeze(1)
        single_point = single_point.unsqueeze(0).repeat(args_infer["img_size"] // 14 * args_infer["img_size"] // 14,
                                                    1).unsqueeze(1)
        
    all_points = torch.cat([single_point, all_points], dim=1).to(device)
    mean_feats = []
    dif = (all_points[:, 0, :] - all_points[:, 1, :]).type(torch.float32).norm(p=2, dim=1)
    dif_n = (dif / mask).unsqueeze(1)
    for i in range(all_points.shape[0]):
        imix = int(all_points[i, :, 0].min().cpu())
        imax = int(all_points[i, :, 0].max().cpu())
        ymix = int(all_points[i, :, 1].min().cpu())
        ymax = int(all_points[i, :, 1].max().cpu())
        features_i = features[imix:imax + 1, ymix:ymax + 1, :]
        # attn_i = attn_norms[imix:imax+1, ymix:ymax+1].mean()
        features_i = features_i.reshape(features_i.shape[0] * features_i.shape[1], features_i.shape[2]).mean(0)
        if i == 0:
            mean_feats = features_i.unsqueeze(0)
        else:
            mean_feats = torch.cat([mean_feats, features_i.unsqueeze(0)], dim=0)
    with torch.no_grad():
        preds = model(mean_feats.to(device), dif_n.to(device))
    return preds.squeeze(), dif_n.squeeze(), single_point, None

def get_predictions (num_grasps, data,constrain_mode = False):
    items = []
    valid_pts_pred, mask_n, device, features = data['valid_pts_pred'], data['mask_n'], data['device'], data['features']
    model_single, args_infer = data['model_single'], data['args_infer']
    heights = data['heights']
    max_dist, min_dist = data['max_dist'], data['min_dist']
    mask = data['mask']
    for point_idx in range(num_grasps):
        if constrain_mode == True : 
            start = time.time()
            preds, diff_n, single_point, single_point_gt = test_single_point_constraint(valid_pts_pred.cpu(), mask_n, device, features, model_single, args_infer, point_idx,constrain_mode=constrain_mode)
            end = time.time() - start
            #print("single poiint time : ", end * 1000)

            th_p = diff_n>max_dist
            th_n = diff_n<min_dist
            th = th_p + th_n
            preds[th] = 0.



            topk=1
            if topk != 0:
                topk, ind = torch.topk(preds, topk)
                conf = preds[ind]
                ind = ind.cpu()
            preds = torch.zeros(6400)
            preds = preds.reshape(args_infer["img_size"]//14, args_infer["img_size"]//14).unsqueeze(0).unsqueeze(0)

            top_x = 0
            top_y = 0 
            if constrain_mode == True :
                top_y , top_x = int(valid_pts_pred[ind,0,0].item()), int(valid_pts_pred[ind,0,1].item()) 
                preds[0,0,top_x,top_y] = 0 
            preds = torch.nn.functional.interpolate(preds, (args_infer["img_size"], args_infer["img_size"]), mode="nearest").squeeze()
            preds = torch.permute(preds, (1, 0))


            zeros = torch.zeros(args_infer["img_size"], args_infer["img_size"], 1)
            ones = torch.ones(args_infer["img_size"], args_infer["img_size"], 1)
            preds = torch.cat([preds.cpu().detach().unsqueeze(2), zeros, zeros], dim = 2)
            origin_point = np.zeros((3, args_infer["img_size"]//14, args_infer["img_size"]//14))
            try : 
                origin_point[:, single_point[0][0][0], single_point[0][0][1]] = [0, 1, 0]
            except : 
                origin_point[:, int(single_point[0][0][0].item()), int(single_point[0][0][1].item())] = [0, 1, 0]
                
            if constrain_mode == True : 
                origin_point[:, top_y, top_x] = [0, 1, 1]
            origin_point = torch.nn.functional.interpolate(torch.tensor(origin_point).unsqueeze(0), (args_infer["img_size"], args_infer["img_size"]), mode="nearest").squeeze()
            
            pred_point = torch.Tensor([top_y, top_x]).to(torch.int64) * 14 + 7
            item = {}
            item['origin_point'] = origin_point
            item['preds'] = preds
            item['single_point'] = single_point[0,0] * 14 + 7
            item['pred_point'] = pred_point
            item['conf'] = conf.item()
            
            items.append(item)
        
        else :    
            single_point = None 
            preds, diff_n, single_point, single_point_gt = test_two_points(valid_pts_pred.reshape(-1,2).cpu(), mask_n, 
                                        device, features, model_single, args_infer, point_idx,single_point=None)
            
            height = sum(heights)/ len(heights)  #height is set to the gknet paper value -> mean of all the heights
            th_p = diff_n>max_dist
            th_n = diff_n<min_dist
            th = th_p + th_n
            preds[th] = 0.
            preds[mask] = 0.
            topk=1
            if topk != 0:
                topk, ind = torch.topk(preds, topk)
                ind = ind.cpu()
                conf = preds[ind]
                preds = torch.zeros(6400)

                preds[ind] = 1
            
            x, y = ind// (args_infer["img_size"]//14), ind % (args_infer["img_size"]//14)
            pred_point = torch.Tensor([y.item(), x.item()]).to(torch.int64) * 14 + 7  ## predicted second point 
            input_point = single_point[0,0] * 14 + 7## single input point prediction => input prediction 
            preds = preds.reshape(args_infer["img_size"]//14, args_infer["img_size"]//14).unsqueeze(0).unsqueeze(0)

            
            preds = torch.nn.functional.interpolate(preds, (args_infer["img_size"], args_infer["img_size"]), mode="nearest").squeeze()
            preds = torch.permute(preds, (1, 0))
            zeros = torch.zeros(args_infer["img_size"], args_infer["img_size"], 1)
            preds = torch.cat([zeros,zeros,preds.cpu().detach().unsqueeze(2)], dim = 2)
            origin_point = np.zeros((3, args_infer["img_size"]//14, args_infer["img_size"]//14))
            try : 
                origin_point[:, single_point[0][0][0], single_point[0][0][1]] = [0, 1, 0]
            except : 
                origin_point[:, int(single_point[0][0][0].item()), int(single_point[0][0][1].item())] = [0, 1, 0]
            #origin_point[:, single_point_gt[0][0][0], single_point_gt[0][0][1]] = [0, 0, 1] gt is uncommented as it just delivers a random point
            
            
            origin_point = torch.nn.functional.interpolate(torch.tensor(origin_point).unsqueeze(0), (args_infer["img_size"], 
                                                    args_infer["img_size"]), mode="nearest").squeeze()
            
            item = {}
            item['origin_point'] = origin_point
            item['preds'] = preds
            item['single_point'] = single_point[0,0] * 14 + 7
            item['pred_point'] = pred_point
            item['conf'] = conf.item()
            
            items.append(item)
    return items 


def get_predictions_left_right(num_grasps, data,constrain_mode = False):
    items = []
    for point_idx in range(num_grasps):
        valid_pts_pred_left, mask_n, device, features = data['valid_pts_pred_left'], data['mask_n'], data['device'], data['features']
        valid_pts_pred_right = data['valid_pts_pred_right']
        model_single, args_infer = data['model_single'], data['args_infer']
        heights = data['heights']
        max_dist, min_dist = data['max_dist'], data['min_dist']
        mask = data['mask']
        
        if constrain_mode == True : 
            preds, diff_n, single_point, single_point_gt = test_single_point_constraint_left_right(valid_pts_pred_left.cpu(), valid_pts_pred_right.cpu(),mask_n, device, features, model_single, args_infer, point_idx,constrain_mode=constrain_mode)
        
            th_p = diff_n>max_dist
            th_n = diff_n<min_dist
            th = th_p + th_n
            preds[th] = 0.



            topk=1
            if topk != 0:
                topk, ind = torch.topk(preds, topk)
                conf = preds[ind]
                ind = ind.cpu()
            preds = torch.zeros(6400)
            preds = preds.reshape(args_infer["img_size"]//14, args_infer["img_size"]//14).unsqueeze(0).unsqueeze(0)

            top_x = 0
            top_y = 0 
            if constrain_mode == True :
                top_y , top_x = int(valid_pts_pred_right[ind,0,0].item()), int(valid_pts_pred_right[ind,0,1].item()) 
                preds[0,0,top_x,top_y] = 0 
            preds = torch.nn.functional.interpolate(preds, (args_infer["img_size"], args_infer["img_size"]), mode="nearest").squeeze()
            preds = torch.permute(preds, (1, 0))


            zeros = torch.zeros(args_infer["img_size"], args_infer["img_size"], 1)
            ones = torch.ones(args_infer["img_size"], args_infer["img_size"], 1)
            preds = torch.cat([preds.cpu().detach().unsqueeze(2), zeros, zeros], dim = 2)
            origin_point = np.zeros((3, args_infer["img_size"]//14, args_infer["img_size"]//14))
            try : 
                origin_point[:, single_point[0][0][0], single_point[0][0][1]] = [0, 1, 0]
            except : 
                origin_point[:, int(single_point[0][0][0].item()), int(single_point[0][0][1].item())] = [0, 1, 0]
                
            if constrain_mode == True : 
                origin_point[:, top_y, top_x] = [0, 1, 1]
            origin_point = torch.nn.functional.interpolate(torch.tensor(origin_point).unsqueeze(0), (args_infer["img_size"], args_infer["img_size"]), mode="nearest").squeeze()
            
            pred_point = torch.Tensor([top_y, top_x]).to(torch.int64) * 14 + 7
            item = {}
            item['origin_point'] = origin_point
            item['preds'] = preds
            item['single_point'] = single_point[0,0] * 14 + 7
            item['pred_point'] = pred_point
            item['conf'] = conf.item()
            
            items.append(item)
        
        else :    
            single_point = None 
            
            preds, diff_n, single_point, single_point_gt = test_two_points(valid_pts_pred_left.reshape(-1,2).cpu(), mask_n, 
                                        device, features, model_single, args_infer, point_idx,single_point=None)
            
            height = sum(heights)/ len(heights)  #height is set to the gknet paper value -> mean of all the heights
            th_p = diff_n>max_dist
            th_n = diff_n<min_dist
            th = th_p + th_n
            preds[th] = 0.
            preds[mask] = 0.
            topk=1
            if topk != 0:
                topk, ind = torch.topk(preds, topk)
                ind = ind.cpu()
                conf = preds[ind]
                preds = torch.zeros(6400)

                preds[ind] = 1
            
            x, y = ind// (args_infer["img_size"]//14), ind % (args_infer["img_size"]//14)
            pred_point = torch.Tensor([y.item(), x.item()]).to(torch.int64) * 14 + 7  ## predicted second point 
            input_point = single_point[0,0] * 14 + 7## single input point prediction => input prediction 
            preds = preds.reshape(args_infer["img_size"]//14, args_infer["img_size"]//14).unsqueeze(0).unsqueeze(0)

            
            preds = torch.nn.functional.interpolate(preds, (args_infer["img_size"], args_infer["img_size"]), mode="nearest").squeeze()
            preds = torch.permute(preds, (1, 0))
            zeros = torch.zeros(args_infer["img_size"], args_infer["img_size"], 1)
            preds = torch.cat([zeros,zeros,preds.cpu().detach().unsqueeze(2)], dim = 2)
            origin_point = np.zeros((3, args_infer["img_size"]//14, args_infer["img_size"]//14))
            try : 
                origin_point[:, single_point[0][0][0], single_point[0][0][1]] = [0, 1, 0]
            except : 
                origin_point[:, int(single_point[0][0][0].item()), int(single_point[0][0][1].item())] = [0, 1, 0]
            #origin_point[:, single_point_gt[0][0][0], single_point_gt[0][0][1]] = [0, 0, 1] gt is uncommented as it just delivers a random point
            
            
            origin_point = torch.nn.functional.interpolate(torch.tensor(origin_point).unsqueeze(0), (args_infer["img_size"], 
                                                    args_infer["img_size"]), mode="nearest").squeeze()
            
            item = {}
            item['origin_point'] = origin_point
            item['preds'] = preds
            item['single_point'] = single_point[0,0] * 14 + 7
            item['pred_point'] = pred_point
            item['conf'] = conf.item()
            
            items.append(item)
    return items 


def vis_preds_with_metrics(num_grasps,items,org_image,grasp,heights,args_infer,preds_cp, topks=10,vis=True): 
    items = sorted(items, key=lambda x: x['conf'], reverse=True)
    top_k_preds = topks if topks <= num_grasps else num_grasps

    total_cnt, correct_cnt = 0, 0
    for i in range(top_k_preds): 
        preds = items[i]['preds']
        origin_point = items[i]['origin_point']
        single_point = items[i]['single_point'] 
        pred_point = items[i]['pred_point'] 
        conf = items[i]['conf']
        correct_end = False
        if vis == True : 
            plt.figure(figsize=(16,16))
        best_iou, best_idx = -1,0
        best_corner_pts = None
        best_corner_preds = None
        best_angle_diff = 1000
        for gt_idx in range(int(grasp.shape[0] // 2.)): 
                    grasp_cur = grasp[gt_idx] * 14 + 7
                    height = sum(heights)/ len(heights)
                    corner_points, corner_points_pred, correct, iou, angle_diff = grasp_correct_full(single_point, pred_point, 
                                                                    grasp_cur,height /2. * args_infer["img_size"] )
                    
                    for pt in grasp_cur : 
                        new_x = int(pt[0]) 
                        new_y = int(pt[1])
                        boarder = 2
                        origin_point[:,new_x - boarder : new_x + boarder , new_y - boarder: new_y + boarder] = torch.zeros((3,boarder*2,boarder*2)) 
                        origin_point[2,new_x - boarder : new_x + boarder , new_y - boarder: new_y + boarder] = torch.ones((boarder*2,boarder*2)) 
                    
                    if correct == True :
                        correct_end = True
                        best_corner_pts = corner_points.clone()
                        best_corner_preds = corner_points_pred.clone()
                        best_iou = iou
                        best_angle_diff = angle_diff
                        break 
                        
                    
                    if correct_end == True :
                        if iou > best_iou and correct == True :
                            best_iou = iou
                            best_idx = gt_idx
                            best_angle_diff = angle_diff
                            best_corner_pts = corner_points.clone()
                            best_corner_preds = corner_points_pred.clone()

                            
                    else : 
                        if iou > best_iou :
                            best_iou = iou
                            best_idx = gt_idx
                            best_angle_diff = angle_diff
                            best_corner_pts = corner_points.clone()
                            best_corner_preds = corner_points_pred.clone()
        if correct_end == True :
            correct_cnt += 1
        total_cnt += 1
        
        if True == True : 
            for pt in best_corner_pts : 
                new_x = int(pt[0]) 
                new_y = int(pt[1])
                boarder = 2
                origin_point[:,new_x - boarder : new_x + boarder , new_y - boarder: new_y + boarder] = torch.zeros((3,boarder*2,boarder*2)) 
                origin_point[1,new_x - boarder : new_x + boarder , new_y - boarder: new_y + boarder] = torch.ones((boarder*2,boarder*2)) 
            
            for pt in best_corner_preds : 
                    new_x = int(pt[0]) 
                    new_y = int(pt[1])
                    boarder = 2
                    origin_point[:,new_x - boarder : new_x + boarder , new_y - boarder: new_y + boarder] = torch.ones((3,boarder*2,boarder*2)) 
                    origin_point[0,new_x - boarder : new_x + boarder , new_y - boarder: new_y + boarder] = torch.zeros((boarder*2,boarder*2)) 
            
            #origin_point[:,single_point[0] - 2 : single_point[0] + 2 , single_point[1] - 2: single_point[1] + 2] = torch.ones((3,4,4)) * 0.5
            #origin_point[:,pred_point[0] - 2 : pred_point[0] + 2 , pred_point[1] - 2: pred_point[1] + 2] = torch.ones((3,4,4)) * 0.5

            origin_point = torch.permute(origin_point,(1, 2, 0)).cpu().detach().numpy() + 0.2*preds_cp.cpu().detach().numpy()
            show_img = org_image + 0.7*preds.cpu().detach().numpy() + 0.7*origin_point
            #show_img = org_image + 0.7*origin_point2 + 0.7*origin_point
            plt.imshow(show_img)
            #plt.title("iou : {} | angle offset : {} degrees |  correct : {} | grasp conf : {}".format(round(best_iou,2),round(best_angle_diff.item(),2)  , correct_end, round(conf,2)))
            #plt.savefig('store_dir/{}.png'.format(i))
            #plt.close()
        
    #print("Accuracy is {} %".format(round(correct_cnt / total_cnt * 100,2)))
    return correct_cnt / total_cnt * 100


def vis_preds_with_metrics_left_right(num_grasps,items,org_image,grasp,heights,args_infer,preds_cp_left,preds_cp_right, topks=10): 
    items = sorted(items, key=lambda x: x['conf'], reverse=True)
    top_k_preds = topks if topks <= num_grasps else num_grasps

    total_cnt, correct_cnt = 0, 0
    for i in range(top_k_preds): 
        preds = items[i]['preds']
        origin_point = items[i]['origin_point']
        single_point = items[i]['single_point'] 
        pred_point = items[i]['pred_point'] 
        conf = items[i]['conf']
        correct_end = False
        plt.figure(figsize=(16,16))
        best_iou, best_idx = -1,0
        best_corner_pts = None
        best_corner_preds = None
        best_angle_diff = 1000
        for gt_idx in range(int(grasp.shape[0] // 2.)): 
                    grasp_cur = grasp[gt_idx] * 14 + 7
                    height = sum(heights)/ len(heights)
                    corner_points, corner_points_pred, correct, iou, angle_diff = grasp_correct_full(single_point, pred_point, 
                                                                    grasp_cur,height /2. * args_infer["img_size"] )
                    
                    for pt in grasp_cur : 
                        new_x = int(pt[0]) 
                        new_y = int(pt[1])
                        boarder = 2
                        origin_point[:,new_x - boarder : new_x + boarder , new_y - boarder: new_y + boarder] = torch.zeros((3,boarder*2,boarder*2)) 
                        origin_point[2,new_x - boarder : new_x + boarder , new_y - boarder: new_y + boarder] = torch.ones((boarder*2,boarder*2)) 
                    
                    if correct == True :
                        correct_end = True
                        best_corner_pts = corner_points.clone()
                        best_corner_preds = corner_points_pred.clone()
                        best_iou = iou
                        best_angle_diff = angle_diff
                        
                    
                    if correct_end == True :
                        if iou > best_iou and correct == True :
                            best_iou = iou
                            best_idx = gt_idx
                            best_angle_diff = angle_diff
                            best_corner_pts = corner_points.clone()
                            best_corner_preds = corner_points_pred.clone()

                            
                    else : 
                        if iou > best_iou :
                            best_iou = iou
                            best_idx = gt_idx
                            best_angle_diff = angle_diff
                            best_corner_pts = corner_points.clone()
                            best_corner_preds = corner_points_pred.clone()

        if correct_end == True :
            correct_cnt += 1
        total_cnt += 1
        
        for pt in best_corner_pts : 
            new_x = int(pt[0]) 
            new_y = int(pt[1])
            boarder = 2
            origin_point[:,new_x - boarder : new_x + boarder , new_y - boarder: new_y + boarder] = torch.zeros((3,boarder*2,boarder*2)) 
            origin_point[1,new_x - boarder : new_x + boarder , new_y - boarder: new_y + boarder] = torch.ones((boarder*2,boarder*2)) 
        
        for pt in best_corner_preds : 
                new_x = int(pt[0]) 
                new_y = int(pt[1])
                boarder = 2
                origin_point[:,new_x - boarder : new_x + boarder , new_y - boarder: new_y + boarder] = torch.ones((3,boarder*2,boarder*2)) 
                origin_point[0,new_x - boarder : new_x + boarder , new_y - boarder: new_y + boarder] = torch.zeros((boarder*2,boarder*2)) 
        
        #origin_point[:,single_point[0] - 2 : single_point[0] + 2 , single_point[1] - 2: single_point[1] + 2] = torch.ones((3,4,4)) * 0.5
        #origin_point[:,pred_point[0] - 2 : pred_point[0] + 2 , pred_point[1] - 2: pred_point[1] + 2] = torch.ones((3,4,4)) * 0.5
        if True == True : 
            origin_point = torch.permute(origin_point,(1, 2, 0)).cpu().detach().numpy() + 0.2*preds_cp_left.cpu().detach().numpy() + 0.2*preds_cp_right.cpu().detach().numpy()
            #show_img = org_image + 0.7*preds.cpu().detach().numpy() + 0.7*origin_point
            show_img = org_image + 0.7*origin_point
            #show_img = org_image + 0.7*origin_point2 + 0.7*origin_point
            plt.imshow(show_img)
            plt.title("iou : {} | angle offset : {} degrees |  correct : {} | grasp conf : {}".format(round(best_iou,2),round(best_angle_diff.item(),2)  , correct_end, round(conf,2)))
            plt.savefig('store_dir/{}.png'.format(i))

    print("Accuracy is {} %".format(round(correct_cnt / total_cnt * 100,2)))
    return correct_cnt / total_cnt * 100
    
 
    
def get_valid_points(all_points, features,model,device='cuda',PATCH_DIM=1120//14,IMAGE_SIZE=1120): 
    mean_feats=[]
    diffs = []
    patch_area = 1
    '''
    for i in range(all_points.shape[0]):
        pt1 = all_points[i,0,:]
                        
        features_1 = features[pt1[0]:pt1[0]+patch_area, pt1[1]:pt1[1]+patch_area, :]
        features_1 = features_1.reshape(features_1.shape[0] * features_1.shape[1], features_1.shape[2]).mean(0)

                    
        if i == 0:
            mean_feats = features_1.unsqueeze(0)
        else:
            mean_feats = torch.cat([mean_feats, features_1.unsqueeze(0)], dim=0)
    
    '''    
    #mean_feats2 = torch.zeros((6400,768)).to(device)
    #features = features.repeat()
    #pt1 = all_points[:,0,:]
    pt1 = all_points[:, 0, :]
    row_indices = pt1[:, 0]
    col_indices = pt1[:, 1]

    patch_indices = torch.arange(patch_area).to(device)
    row_indices = row_indices[:, None] + patch_indices
    col_indices = col_indices[:, None] + patch_indices

    features_1 = features[row_indices, col_indices].reshape(-1, patch_area, 768)
    mean_feats = features_1.mean(1)
    
    with torch.no_grad():
        preds = model.forward_valid(mean_feats.to(device))

    preds = preds.squeeze().reshape(PATCH_DIM, PATCH_DIM).unsqueeze(0).unsqueeze(0)
    preds_patches = preds
    zeros = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, 1)
    ones = torch.ones(IMAGE_SIZE, IMAGE_SIZE, 1)
    preds = torch.nn.functional.interpolate(preds, (IMAGE_SIZE, IMAGE_SIZE), mode="nearest").squeeze()
    preds = torch.permute(preds, (1, 0))
    preds = torch.cat([preds.cpu().detach().unsqueeze(2), zeros, zeros], dim = 2)
    return preds, preds_patches


def get_valid_points_left_right(all_points, features,model,device='cuda',PATCH_DIM=1120//14,IMAGE_SIZE=1120): 
    mean_feats=[]
    diffs = []
    patch_area = 1
    for i in range(all_points.shape[0]):
        pt1 = all_points[i,0,:]
                        
        features_1 = features[pt1[0]:pt1[0]+patch_area, pt1[1]:pt1[1]+patch_area, :]
        features_1 = features_1.reshape(features_1.shape[0] * features_1.shape[1], features_1.shape[2]).mean(0)

                    
        if i == 0:
            mean_feats = features_1.unsqueeze(0)
        else:
            mean_feats = torch.cat([mean_feats, features_1.unsqueeze(0)], dim=0)
    with torch.no_grad():
        preds = model.forward_valid(mean_feats.to(device))

    preds_left = preds[:,1].squeeze().reshape(PATCH_DIM, PATCH_DIM).unsqueeze(0).unsqueeze(0)
    preds_right = preds[:,2].squeeze().reshape(PATCH_DIM, PATCH_DIM).unsqueeze(0).unsqueeze(0)
    preds_patches_left = preds_left
    preds_patches_right = preds_right
    zeros = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, 1)
    ones = torch.ones(IMAGE_SIZE, IMAGE_SIZE, 1)
    preds_left = torch.nn.functional.interpolate(preds_left, (IMAGE_SIZE, IMAGE_SIZE), mode="nearest").squeeze()
    preds_left = torch.permute(preds_left, (1, 0))
    preds_left = torch.cat([preds_left.cpu().detach().unsqueeze(2), zeros, zeros], dim = 2)
    
    preds_right = torch.nn.functional.interpolate(preds_right, (IMAGE_SIZE, IMAGE_SIZE), mode="nearest").squeeze()
    preds_right = torch.permute(preds_right, (1, 0))
    preds_right = torch.cat([preds_right.cpu().detach().unsqueeze(2), zeros, zeros], dim = 2)
    return preds_left, preds_right, preds_patches_left, preds_patches_right

def get_topk_valid_points(preds,preds_patches, mask_i, topk_num=10,TOP_K=True,device='cuda'):
    pts = None
    if TOP_K : 
        preds_cp = preds_patches
        mask_i = torch.nn.functional.interpolate(mask_i.unsqueeze(0), (80, 80), mode="bilinear").squeeze().flatten()
        k = topk_num
        flattened_tensor = preds_cp.flatten()
        flattened_tensor[mask_i>0.2] = 0.
        # Find the top k values and their indices
        top_values, top_indices = torch.topk(flattened_tensor, k)
        # Create a mask tensor where the top k values are True and the rest are False
        mask = torch.zeros_like(flattened_tensor)
        mask[top_indices] = 1.
        mask = mask.reshape(preds_cp.shape)
        mask = mask.permute(0,1,3,2)
        pts = (mask == 1).nonzero() * 14 + 7
        pts = pts[:, 2:4]
        pts = torch.cat([pts, torch.zeros(pts.shape[0], 1).to(device)], dim=1)
        mask = torch.nn.functional.interpolate(mask, (IMAGE_SIZE, IMAGE_SIZE), mode="nearest").squeeze()
        zeros = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, 1)
        preds_cp = torch.cat([mask.unsqueeze(2),zeros.to(device), zeros.to(device)], dim = 2)

    else : 
        preds_cp = preds.clone()
        thresh = 0.98
        preds_cp[preds_cp < thresh] = 0
        preds_cp[preds_cp >= thresh] = 1
        pts = (preds_cp == 1).nonzero() 
    
    valid_pts_pred = pts[:, 0:2] // 14
    valid_pts_pred = torch.unique(valid_pts_pred, dim=0)
    valid_pts_pred = valid_pts_pred.reshape(-1, 1,2) 
    return preds_cp, pts, valid_pts_pred


def get_topk_valid_points_left_right(preds_left,preds_right,preds_patches_left,preds_patches_right, mask_i, topk_num=10,TOP_K=True,device='cuda'):
    pts = None
    if TOP_K : 
        preds_cp_left = preds_patches_left
        mask_i = torch.nn.functional.interpolate(mask_i.unsqueeze(0), (80, 80), mode="bilinear").squeeze().flatten()
        k = topk_num
        flattened_tensor_left = preds_cp_left.flatten()
        flattened_tensor_left[mask_i>0.2] = 0.
        # Find the top k values and their indices
        top_values, top_indices = torch.topk(flattened_tensor_left, k)
        # Create a mask tensor where the top k values are True and the rest are False
        mask_left = torch.zeros_like(flattened_tensor_left)
        mask_left[top_indices] = 1.
        mask_left = mask_left.reshape(preds_cp_left.shape)
        mask_left = mask_left.permute(0,1,3,2)
        pts_left = (mask_left == 1).nonzero() * 14 + 7
        pts_left = pts_left[:, 2:4]
        pts_left = torch.cat([pts_left, torch.zeros(pts_left.shape[0], 1).to(device)], dim=1)
        mask_left = torch.nn.functional.interpolate(mask_left, (IMAGE_SIZE, IMAGE_SIZE), mode="nearest").squeeze()
        zeros = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, 1)
        preds_cp_left = torch.cat([mask_left.unsqueeze(2),zeros.to(device), zeros.to(device)], dim = 2)
        
        preds_cp_right = preds_patches_right
        flattened_tensor_right = preds_cp_right.flatten()
        flattened_tensor_right[mask_i>0.2] = 0.
        # Find the top k values and their indices
        top_values, top_indices = torch.topk(flattened_tensor_right, k)
        # Create a mask tensor where the top k values are True and the rest are False
        mask_right = torch.zeros_like(flattened_tensor_right)
        mask_right[top_indices] = 1.
        mask_right = mask_right.reshape(preds_cp_right.shape)
        mask_right = mask_right.permute(0,1,3,2)
        pts_right = (mask_right == 1).nonzero() * 14 + 7
        pts_right = pts_right[:, 2:4]
        pts_right = torch.cat([pts_right, torch.zeros(pts_right.shape[0], 1).to(device)], dim=1)
        mask_right = torch.nn.functional.interpolate(mask_right, (IMAGE_SIZE, IMAGE_SIZE), mode="nearest").squeeze()
        zeros = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, 1)
        preds_cp_right = torch.cat([zeros.to(device), mask_right.unsqueeze(2),zeros.to(device)], dim = 2)

    else : 
        preds_cp = preds_left.clone()
        thresh = 0.98
        preds_cp[preds_cp < thresh] = 0
        preds_cp[preds_cp >= thresh] = 1
        pts = (preds_cp == 1).nonzero() 
    
    valid_pts_pred_left = pts_left[:, 0:2] // 14
    valid_pts_pred_left = torch.unique(valid_pts_pred_left, dim=0)
    valid_pts_pred_left = valid_pts_pred_left.reshape(-1, 1,2) 
    
    
    valid_pts_pred_right = pts_right[:, 0:2] // 14
    valid_pts_pred_right = torch.unique(valid_pts_pred_right, dim=0)
    valid_pts_pred_right = valid_pts_pred_right.reshape(-1, 1,2) 
    return preds_cp_left, pts_left, valid_pts_pred_left, preds_cp_right, pts_right, valid_pts_pred_right

def visualize_valid_points(grasp,mask,org_image,preds_cp,IMAGE_SIZE=1120): 
    PATCH_DIM = IMAGE_SIZE//14
    grasp_vis = torch.zeros((PATCH_DIM, PATCH_DIM))
    for g in grasp:
            g1, g2 = g[0], g[1]
            grasp_vis[g1[0], g1[1]] = 1
            grasp_vis[g2[0], g2[1]] = 1
    grasp_vis = grasp_vis.unsqueeze(0).unsqueeze(0)
    grasp_vis = torch.nn.functional.interpolate(grasp_vis, (IMAGE_SIZE, IMAGE_SIZE), mode="nearest").squeeze()
    zeros = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, 1)
    grasp_vis = torch.cat([zeros,grasp_vis.cpu().detach().unsqueeze(2), zeros], dim = 2)
    
    mask_vis = torch.zeros((PATCH_DIM, PATCH_DIM))
    mask = mask.permute(0,2,1)
    zero_indices = torch.nonzero(mask[0] == 0) // 14
    one_indices = torch.nonzero(mask[0] == 1)  // 14

    for idcs in one_indices:
        mask_vis[idcs[0], idcs[1]] = 1
    mask_vis = mask_vis.unsqueeze(0).unsqueeze(0)
    mask_vis = torch.nn.functional.interpolate(mask_vis, (IMAGE_SIZE, IMAGE_SIZE), mode="nearest").squeeze()
    zeros = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, 1)
    mask_vis = torch.cat([mask_vis.cpu().detach().unsqueeze(2), zeros, zeros], dim = 2)
    
    #origin_point = np.zeros((3, 60, 60))
    #origin_point[:, single_point[0][0][0], single_point[0][0][1]] = [0, 1, 0]
    #origin_point = torch.nn.functional.interpolate(torch.tensor(origin_point).unsqueeze(0), (840, 840), mode="nearest").squeeze()
    #origin_point = torch.permute(origin_point,(1, 2, 0)).cpu().detach().numpy()
    plt.figure(figsize=(16,16))
    show_img = org_image + 0.7*preds_cp.cpu().detach().numpy()# + 0.5 * grasp_vis.numpy() 
    #show_img = org_image + 0.5 * grasp_vis.numpy() + 0.5 * mask_vis.numpy()
    #show_img = org_image + 0.7*origin_point2 + 0.7*origin_point
    plt.imshow(show_img)
    
def get_second_point_data(dataset,data_len,model_single,device,args_infer,inv_transform,test_idx=0): 
    img, mask, grasp,heights,corners = get_features(dataset[test_idx], model_single, device, args_infer)
    org_image = torch.permute(inv_transform(img), (1, 2, 0)).cpu().numpy()
    mask_n = mask.sum().sqrt()
    mask = torch.nn.functional.interpolate(mask.unsqueeze(0), (args_infer["img_size"]//14, args_infer["img_size"]//14), mode="nearest").squeeze()
    mask = mask.reshape((args_infer["img_size"]//14)**2)
    mask = mask>0
    return mask, mask_n,  grasp, heights, corners
