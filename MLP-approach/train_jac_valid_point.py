import torch
import numpy as np
from dataset_jacquard_samples import JacquardSamples
from utils import get_transform, augment_image
from bce_model import BCEGraspTransformer
from utils_train import create_correct_false_points_mask
import random
from pytorch_lightning.loggers import TensorBoardLogger

def train(dataset, model, args_train, device):
    print("train")
    params = [
        {
            'params': model.linear_head.parameters(),
            'lr': args_train["lr"]
        },
        {
            'params': model.linear_head2.parameters(),
            'lr': args_train["lr"]
        }

    ]
    optim = torch.optim.Adam(params)
    loss_bce = torch.nn.BCELoss()
    train_loss_running = 0.
    model.to(device)
    logger = TensorBoardLogger("logs", name=args_train['experiment_name'])
    IMAGE_SIZE = 1120 
    PATCH_DIM = IMAGE_SIZE // 14
    iter = 0.
    tot_iter = 0
    for epoch in range(args_train["num_epochs"]):
        for i in range(len(dataset)):
            optim.zero_grad()
            data = dataset[i]
            mask = data["mask"].to(device)
            img = data["img"].to(device)
            img = torch.permute(img, (0, 2, 1))
            grasp = data["points_grasp"]//14
            grasp = grasp.to(device)
            false_points, left_grasps, right_grasps = create_correct_false_points_mask(grasp, args_train["batch_size"],mask,img,VIS=False)
            bs = args_train['batch_size'] *2  #batch size
            div_bs = bs // 3
            remaining_bs = bs - div_bs * 2
            false_points = false_points[:remaining_bs]
            try : 
                left_grasps = left_grasps.reshape(-1, 2, 2)
                right_grasps = right_grasps.reshape(-1, 2, 2)
            except :
                left_grasps = left_grasps[:left_grasps.shape[0]-1].reshape(-1, 2, 2)
                right_grasps = right_grasps[:right_grasps.shape[0]-1].reshape(-1, 2, 2)

            
            left_grasps = left_grasps[:div_bs]
            right_grasps = right_grasps[:div_bs]
            
            idx = random.sample(range(grasp.shape[0]), args_train["batch_size"])
            all_points = torch.cat([left_grasps, right_grasps,false_points], dim=0).to(device)
            #all_points = torch.cat([grasp[idx], false_points], dim=0).to(device)
            features, _ = model.forward_dino_features(img.unsqueeze(0))
            features = features.squeeze().reshape(PATCH_DIM, PATCH_DIM, 768)
            mean_feats=[]
            patch_area = 1
            for i in range(all_points.shape[0]):
                pt1 = all_points[i,0,:]
                pt2 = all_points[i,1,:]
                
                x,y = int(pt1[0].item()), int(pt1[1].item())
                features_1 = features[x:x+patch_area, y:y+patch_area, :]
                features_1 = features_1.reshape(features_1.shape[0] * features_1.shape[1], features_1.shape[2]).mean(0)
                
                x,y = int(pt2[0].item()), int(pt2[1].item()) 
                features_2 = features[x:x+patch_area, y:y+patch_area, :]
                features_2 = features_2.reshape(features_2.shape[0] * features_2.shape[1], features_2.shape[2]).mean(0)
                
                
                if i == 0:
                    mean_feats = features_1.unsqueeze(0)
                    mean_feats = torch.cat([mean_feats, features_2.unsqueeze(0)], dim=0)
                else:
                    mean_feats = torch.cat([mean_feats, features_1.unsqueeze(0)], dim=0)
                    mean_feats = torch.cat([mean_feats, features_2.unsqueeze(0)], dim=0)

            #gt = torch.cat([torch.ones(2*args_train["batch_size"]), torch.zeros(2*args_train["batch_size"])]).to(device)
            gt = torch.cat([torch.ones(div_bs * 2), torch.ones(div_bs * 2) * 2., torch.zeros(remaining_bs * 2)]).to(device).to(torch.int64)
            
            pred = model.forward_valid(mean_feats)
            target_one_hot = torch.eye(3)[gt]
            loss = loss_bce(pred, target_one_hot)
            loss.backward()
            optim.step()


            train_loss_running += loss.item()
            tot_iter = tot_iter + 1
            iter = iter + 1
            if iter == args_train["print_every_n"]:
                print(f"[{tot_iter}] train_loss: {train_loss_running / args_train['print_every_n']:.6f}")
                iter = 0

                logger.log_metrics({
                    'train_loss': train_loss_running / args_train['print_every_n'],
                    'iter': tot_iter
                }, tot_iter)
                train_loss_running = 0.

    torch.save(model.state_dict(), f'runs/{args_train["experiment_name"]}.ckpt')


def main(args_train):
    device = torch.device(args_train["device"]) if torch.cuda.is_available() else torch.device("cpu")
    image_transform = get_transform()
    model = BCEGraspTransformer(img_size=args_train['img_size'],int_dim=256,output_dim=128)
    dataset = JacquardSamples(dataset_root=args_train["split"], image_transform=image_transform, num_targets=5,
                              overfit=False,
                              img_size=args_train["img_size"], idx=args_train["num_objects"])
    train(dataset, model, args_train, device)
