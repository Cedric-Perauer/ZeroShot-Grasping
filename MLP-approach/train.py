#from train_jac_single import main
from train_jac_unet import main


args_train = {
    "split" : "Bottle_train/",
    "device": "cuda",
    "angle_mode": False,
    "img_size": 1120,
    "num_epochs": 200,
    "num_objects": 1,
    "lr": 1e-3,
    "batch_size": 64,
    "print_every_n": 1,
    "experiment_name": "bottle_1_double",
    'rgb': True,
    'dino_feats': False,
}

main(args_train)