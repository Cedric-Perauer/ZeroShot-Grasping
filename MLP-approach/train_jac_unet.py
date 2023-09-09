import torch
import numpy as np
from dataset_jacquard_samples import JacquardSamples
from utils import get_transform, augment_image
from bce_model import BCEGraspTransformer
from utils_train import create_correct_false_points, create_correct_false_grasps_mask, create_unet_mask
import random
import torch.nn.functional as F
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
        for i in range(len(dataset)):
            optim.zero_grad()
            data = dataset[i]
            img = data["img"].to(device)
            height = data['height']
            img = torch.permute(img, (0, 2, 1))
            mask = data["mask"].sum().sqrt().to(device)
            obj_mask = data['mask'].to(device)  
            grasp = data["points_grasp"]//14
            grasp_inv = torch.cat([grasp[:,1,:].unsqueeze(1), grasp[:,0,:].unsqueeze(1)], dim=1)
            grasp = torch.cat([grasp, grasp_inv], dim=0)
            #false_points = create_correct_false_points(grasp, args_train["batch_size"])
            #idx = random.sample(range(grasp.shape[0]), args_train["batch_size"])
            create_unet_mask(data['resized_mask'].to(device),grasp)
            
            #pred = model(mean_feats, dif_n).squeeze()
            #loss = loss_bce(pred, gt)
            #loss.backward()
            #optim.step()


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
    model = BCEGraspTransformer(img_size=args_train["img_size"])
    dataset = JacquardSamples(dataset_root= args_train["split"] ,image_transform=image_transform, num_targets=5, overfit=False,
                              img_size=args_train["img_size"], idx=args_train["num_objects"])
    print(len(dataset))
    device = args_train["device"] if torch.cuda.is_available() else "cpu"
    train(dataset, model, args_train, device)
