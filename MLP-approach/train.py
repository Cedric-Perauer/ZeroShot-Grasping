from train_jac_single import main


args_train = {
    "device": "cuda",
    "angle_mode": False,
    "img_size": 1120,
    "num_epochs": 500,
    "num_images": 4,
    "lr": 1e-3,
    "batch_size": 64,
    "print_every_n": 1,
    "experiment_name": "bce_grasp_bottle_mask4"
}

main(args_train)