import torch
import numpy as np
from dataset_jacquard_samples import JacquardSamples
from utils import get_transform, augment_image
from model import GraspTransformer


def train(dataset, model, args_train, device):
    optim = torch.optim.Adam(model.parameters(), lr=args_train["lr"])
    loss = torch.nn.BCELoss()
    for epoch in range(args_train["num_epochs"]):
        for i in range(args_train["num_images"]):
            data = dataset[i]
            img = data["img"]
            #grasp points
            grasp = data["points_grasp"]
            grasps_true = grasp[np.random.choice(127, 32)]
            grasps_true = grasps_true//14
            x_mean = torch.tensor(grasp[:,:,0], dtype=torch.float).mean()
            x_std = torch.tensor(grasp[:,:,0], dtype=torch.float).std()
            y_mean = grasp[:,:,1]









def main(args_train):
    device = torch.device(args_train["device"])
    image_transform = get_transform()
    model = GraspTransformer(angle_mode=args_train["angle_mode"], img_size=args_train["img_size"], SIM=True)
    dataset = JacquardSamples(image_transform=image_transform, num_targets=5, overfit=False,
                              img_size=args_train["img_size"])
    train(dataset, model, args_train, device)
