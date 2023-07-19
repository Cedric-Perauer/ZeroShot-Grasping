from train_jac_single import main


args_train = {
    "split" : r"Bottle_test/",
    "device": "cuda",
    "angle_mode": False,
    "img_size": 1120,
    "num_epochs": 0,
    "num_objects": 2,
    "lr": 5e-4,
    "batch_size": 64,
    "print_every_n": 1,
    "experiment_name": "bottle_1"
}

main(args_train)