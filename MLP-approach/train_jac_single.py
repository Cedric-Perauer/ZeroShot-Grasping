import torch
import numpy as np
from dataset_jacquard_samples import JacquardSamples
from utils import get_transform, augment_image
from bce_model import BCEGraspTransformer
from utils_train import create_correct_false_points
import random
from pytorch_lightning.loggers import TensorBoardLogger

def train(dataset, model, args_train, device):
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
            diffs = []
            for i in range(all_points.shape[0]):
                imix = all_points[i,:,0].min()
                imax = all_points[i,:,0].max()
                ymix = all_points[i,:,1].min()
                ymax = all_points[i,:,1].max()
                dif = (all_points[i, 0, :] - all_points[i, 1, :]).type(torch.float32)
                if dif[0] == 0 and dif[1] == 0:
                    dif = torch.zeros(4).to(device)
                else:
                    dif = dif / dif.norm(p=2, dim=-1, keepdim=True)
                    dif = torch.cat([dif, dif*-1])
                features_i = features[imix:imax+1, ymix:ymax+1, :]
                features_i = features_i.reshape(features_i.shape[0] * features_i.shape[1], features_i.shape[2]).mean(0)
                if i == 0:
                    mean_feats = features_i.unsqueeze(0)
                    diffs = dif.unsqueeze(0)
                else:
                    mean_feats = torch.cat([mean_feats, features_i.unsqueeze(0)], dim=0)
                    diffs = torch.cat([diffs, dif.unsqueeze(0)], dim=0)

            gt = torch.cat([torch.ones(args_train["batch_size"]), torch.zeros(args_train["batch_size"])]).to(device)
            pred = model(mean_feats, diffs).squeeze()
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
    torch.save(model.state_dict(), f'runs/{args_train["experiment_name"]}.ckpt')
















def main(args_train):
    device = torch.device(args_train["device"])
    image_transform = get_transform()
    model = BCEGraspTransformer(img_size=840)
    dataset = JacquardSamples(image_transform=image_transform, num_targets=5, overfit=False,
                              img_size=args_train["img_size"])
    train(dataset, model, args_train, device)
