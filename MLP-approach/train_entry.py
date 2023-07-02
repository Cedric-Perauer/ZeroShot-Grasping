from train_jac_valid_point import main

args_train = {
    "device": "cuda",
    "angle_mode": False,
    "img_size": 1120,
    "num_epochs": 1000,
    "num_images": 2,
    "lr": 1e-3,
    "batch_size": 64,
    "print_every_n": 1,
    'save_every' : 100,
    "experiment_name": "bce_grasp_entry"
}

main(args_train)