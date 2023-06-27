import torch
import numpy as np
from dataset_jacquard_samples import JacquardSamples
from utils import get_transform, augment_image
from bce_model import BCEGraspTransformer
from utils_train import create_correct_false_points
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
    iter = 0.
    tot_iter = 0
    for epoch in range(args_train["num_epochs"]):
        for i in range(args_train["num_images"]):
            optim.zero_grad()
            data = dataset[i]
            img = data["img"].to(device)
            img = torch.permute(img, (0, 2, 1))
            grasp = data["points_grasp"]//14
            grasp_inv = torch.cat([grasp[:,1,:].unsqueeze(1), grasp[:,0,:].unsqueeze(1)], dim=1)
            grasp = torch.cat([grasp, grasp_inv], dim=0)
            false_points = create_correct_false_points(grasp, args_train["batch_size"])
            idx = random.sample(range(grasp.shape[0]), args_train["batch_size"])
            all_points = torch.cat([grasp[idx], false_points], dim=0).to(device)
            features = model.forward_dino_features(img.unsqueeze(0)).squeeze().reshape(60, 60, 384)
            mean_feats=[]
            patch_area = 1
            for i in range(all_points.shape[0]):
                pt1 = all_points[i,0,:]
                pt2 = all_points[i,1,:]
                                
                features_1 = features[pt1[0]:pt1[0]+patch_area, pt1[1]:pt1[1]+patch_area, :]
                features_1 = features_1.reshape(features_1.shape[0] * features_1.shape[1], features_1.shape[2]).mean(0)
                
                features_2 = features[pt2[0]:pt2[0]+patch_area, pt2[1]:pt2[1]+patch_area, :]
                features_2 = features_2.reshape(features_2.shape[0] * features_2.shape[1], features_2.shape[2]).mean(0)
                
                
                if i == 0:
                    mean_feats = features_1.unsqueeze(0)
                    mean_feats = torch.cat([mean_feats, features_2.unsqueeze(0)], dim=0)
                else:
                    mean_feats = torch.cat([mean_feats, features_1.unsqueeze(0)], dim=0)
                    mean_feats = torch.cat([mean_feats, features_2.unsqueeze(0)], dim=0)

            gt = torch.cat([torch.ones(2*args_train["batch_size"]), torch.zeros(2*args_train["batch_size"])]).to(device)
            pred = model.forward_valid(mean_feats).squeeze()
            loss = loss_bce(pred, gt)
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
    torch.save(model.state_dict(), f'runs/grasp_valid.ckpt')


def main(args_train):
    device = torch.device(args_train["device"])
    image_transform = get_transform()
    model = BCEGraspTransformer(img_size=args_train['img_size'],int_dim=256,output_dim=128)
    dataset = JacquardSamples(image_transform=image_transform, num_targets=5, overfit=False,
                              img_size=args_train["img_size"])
    train(dataset, model, args_train, device)
