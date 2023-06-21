from train_jac_single import main


args_train = {
    "device": "cuda",
    "angle_mode": False,
    "img_size": 840,
    "num_epochs": 1,
    "num_images": 1,
    "lr": 1e-3
}

main(args_train)