from train_jac_valid_point import main

args_train = {
    "split" : "Figure_train/",
    "device": "cuda",
    "angle_mode": False,
    "img_size": 1120,
    "num_epochs": 50,
    "num_objects": 1,
    "lr": 5e-4,
    "batch_size": 32,
    "print_every_n": 1,
    "experiment_name": "figure_1_single_left_right" 
}

main(args_train)