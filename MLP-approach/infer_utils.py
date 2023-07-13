import torch
import numpy

def get_features(dataset, model, device, args_infer, test_idx):
    data = dataset[test_idx]
    img = data["img"].to(device)
    img = torch.permute(img, (0, 2, 1))
    grasp = data["points_grasp"] // 14
    mask = data["mask"]
    height = data['height']
    corners = data['corners']
    grasp_inv = torch.cat([grasp[:, 1, :].unsqueeze(1), grasp[:, 0, :].unsqueeze(1)], dim=1)
    grasp = torch.cat([grasp, grasp_inv], dim=0)
    features, clk = model.forward_dino_features(img.unsqueeze(0))

    features = features.squeeze().reshape(args_infer["img_size"] // 14, args_infer["img_size"] // 14, 384)

    return img, mask, grasp, features, height, corners

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
    return preds.squeeze(), dif_n.squeeze(), single_point, None