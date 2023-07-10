from train_jac_valid_point import main

args_train = {
    "device": "cuda",
    "angle_mode": False,
    "img_size": 1120,
    "num_epochs": 301,
    "num_images": 4,
    "lr": 1e-4,
    "batch_size": 32,
    "print_every_n": 1,
    'save_every' : 100,
    "experiment_name": "bce_grasp_bottle_entry"
}

main(args_train)