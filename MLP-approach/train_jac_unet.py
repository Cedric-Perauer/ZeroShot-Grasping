import torch
import numpy as np
from dataset_jacquard_samples import JacquardSamples
from utils import get_transform, augment_image
from bce_model import BCEGraspTransformer
from utils_train import create_correct_false_points, create_correct_false_grasps_mask, create_unet_mask
import random
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from unet import UNet
import torch.nn as nn
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)



class CustomLoss(nn.Module):
    def __init__(self, weight=None,batch_size=64):
        super(CustomLoss, self).__init__()
        self.weight_dark =  1 / (80 * 79.) 
        self.weight_light = 1 - self.weight_dark
        self.weight_map = torch.ones((batch_size,1,80,80)).to('cuda:0') * self.weight_dark

    def forward(self, pred, target,gt_coords):
        # Calculate your custom loss here
        self.weight_map = torch.ones((gt_coords.shape[0],1,80,80)).to('cuda:0') * self.weight_dark
        gt_coords = gt_coords * 80
        total_loss = 0.0
        for i in range(gt_coords.shape[0]):
            self.weight_map[i,0,int(gt_coords[i,0,0]),int(gt_coords[i,0,1])] = self.weight_light
            #breakpoint()
        
        for i in range(gt_coords.shape[0]):
            loss_ce = F.binary_cross_entropy(F.sigmoid(pred[i,0]), target[i,0])  * self.weight_map[i]
            # loss = (torch.abs(pred - target))  
            #breakpoint()
            loss_sum = loss_ce[0].sum(0).sum(0).reshape(1,)
            total_loss += loss_sum
            
        return total_loss

def flat_softmax(inp,size=80):
        flat = inp.view(-1, size * size)
        flat = torch.nn.functional.softmax(flat, 1)
        return flat.view(-1, 1, size, size)

def soft_argmax(inp,size=80):
        values_y = torch.linspace(0, (size - 1.) / size, size, dtype=inp.dtype, device=inp.device)
        values_x = torch.linspace(0, (size - 1.) / size, size, dtype=inp.dtype, device=inp.device)
        exp_y = (inp.sum(3) * values_y).sum(-1)
        exp_x = (inp.sum(2) * values_x).sum(-1)
        return torch.stack([exp_x, exp_y], -1)


def vis_image(gt_mask,x,y): 
    img = np.zeros((80,80,3))
    img[:,:,0] = gt_mask
    #breakpoint()
    img[x,y,2] = 1.0 
    
    plt.imshow(img)
    plt.show()
    
    

def train(dataset, args_train, device):
    print("Training UNet Model ")
    print(args_train)
    n_channels = 2 
    if args_train['rgb']:
        n_channels += 3
    model = UNet(n_channels=n_channels,n_classes=1)
    
    #l1_loss = nn.L1Loss()
    l1_loss = CustomLoss()
    
    model.train()
    
    params = [
        {
            'params':model.parameters(),
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
            grasp = torch.cat([grasp, grasp_inv], dim=0).to(device)
            #false_points = create_correct_false_points(grasp, args_train["batch_size"])
            #idx = random.sample(range(grasp.shape[0]), args_train["batch_size"])
            input_mask,gt_mask, gt_coords  = create_unet_mask(data['resized_mask'].to(device),grasp,args_train)
            model_input = input_mask.to(device)
            gt_mask = gt_mask.to(device)
            if args_train['rgb']:
                rgb_data = data['resized_img'].to(device).unsqueeze(0)
                rgb_data = rgb_data.repeat(64, 1, 1,1)
                model_input = torch.cat([model_input,rgb_data], dim=1)
            predicted_mask = model(model_input)
            
            
            ## just for vis reasons 
            #hm = flat_softmax(predicted_mask)
            #pred = soft_argmax(hm)
            
            #print('pred',pred)
            #print('rows,cols',row_index,col_index)
            #print('gt_coords',gt_coords * 80)
            if epoch  > 180 : 
                max_indices = torch.argmax(predicted_mask[0,0])
                row_index = max_indices//80
                col_index = max_indices%80
                vis_image(gt_mask[0].cpu().numpy(),row_index,col_index)
                
            
            
            
            
            #loss = l1_loss(pred[0,0],gt_coords[0,0])
            loss = l1_loss(predicted_mask,gt_mask,gt_coords)
            
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
    
    torch.save(model.state_dict(), f'runs/{args_train["experiment_name"]}unet.ckpt')


def main(args_train):
    device = torch.device(args_train["device"])
    image_transform = get_transform()
    dataset = JacquardSamples(dataset_root= args_train["split"] ,image_transform=image_transform, num_targets=5, overfit=False,
                              img_size=args_train["img_size"], idx=args_train["num_objects"])
    print(len(dataset))
    device = args_train["device"] if torch.cuda.is_available() else "cpu"
    train(dataset, args_train, device)
