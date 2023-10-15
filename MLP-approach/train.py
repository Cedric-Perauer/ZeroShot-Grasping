#from train_jac_single import main
from train_jac_unet import main


args_train = {
    "split" : "Objects_train/",
    "device": "cuda",
    "angle_mode": False,
    "img_size": 1120,
    "num_epochs": 300,
    "num_objects": 1,
    "lr": 1e-3,
    "batch_size": 64,
    "print_every_n": 1,
    "experiment_name": "objects_1_double",
    'checkpoint': 'runs/objects_1_double_unet.ckpt',
    'resume': False,
    'rgb': False,
    'dino_feats': False,
}

main(args_train)