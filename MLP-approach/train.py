from train_jac_single import main


args_train = {
    "device": "cuda",
    "angle_mode": False,
    "img_size": 840,
    "num_epochs": 1000,
    "num_images": 2,
    "lr": 1e-3,
    "batch_size": 64,
    "print_every_n": 1,
    "experiment_name": "bce_grasp_single"
}

main(args_train)