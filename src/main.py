import argparse
import os
import shutil
import json 

import torch

from yolo.trainer import MOD_YOLOTrainer
from dataset.road_r import ROAD_R
from dataset.waymo_road import ROAD_PP

def getArgs():
    parser = argparse.ArgumentParser(description='MOD-CL')
    parser.add_argument("-c", "--cuda", type=str, default="cuda:0", help="CUDA device")
    parser.add_argument("-task", "--task", type=int, default=1, choices=[0, 1, 2, 3, 4], help="Task number")
    parser.add_argument("-dataset", "--dataset", type=str, default="road-r", help="Dataset to use")

    parser.add_argument("-dataset_path", "--dataset_path", type=str, default="../../road-dataset", help="Path to dataset")
    parser.add_argument("-basemodel", "--basemodel", type=str, default="yolov8n", help="YOLO base model")
    parser.add_argument("-max_epochs", "--max_epochs", type=int, default=300, help="Maximum number of epochs")
    parser.add_argument("-req_loss", "--req_loss", type=float, default=0.1, help="Required loss")
    parser.add_argument("-val_split", "--val_split", type=float, default=0.2, help="Validation split")
    parser.add_argument("-seed", "--seed", type=int, default=1, help="Random seed")
    parser.add_argument("-remake", "--remake", action="store_true", help="Remake dataset")
    parser.add_argument("-workers", "--workers", type=int, default=16, help="Number of workers")

    parser.add_argument("-optimizer", "--optimizer", type=str, default="Adam", help="Optimizer to use")
    parser.add_argument("-lr", "--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("-lrf", type=float, default=0.01, help="Learning rate final")
    parser.add_argument("-vanilla", "--vanilla", action="store_true", help="Use vanilla YOLO")
    parser.add_argument("-freeze", "--freeze", type=int, default=0, help="Freeze layers")
    parser.add_argument("-req-type", "--req-type", type=str, default="product", help="Requirements Loss type")
    parser.add_argument("-no_augment", "--no_augment", action="store_true", help="Use Augmentation")
    parser.add_argument("-max_det", "--max_det", type=int, default=300, help="Maximum detections")
    parser.add_argument("-beta_rl", type=float, default=0.1, help="Beta for RL Loss")
    parser.add_argument("-delta_rl", type=float, default=0.5, help="Delta for RL Loss")
    parser.add_argument("-rl_mode", "--rl_mode", type=str, default="normal", help="Requirements Loss type")
    parser.add_argument("-rl", "--reinforcement-loss", action="store_true", help="Use Reinforcement Learning Loss")
    parser.add_argument("-rnd", "--req_num_detect", type=int, default=64, help="Number of required detections")
    parser.add_argument("-rs", "--req_scheduler", type=float, default=0, help="Scheduler for required detections")

    return parser.parse_args()


def on_train_epoch_end(trainer):
    """Callback function to be executed at the end of each training epoch."""
    # Epochs start at 0
    saveFile(trainer)
    hyp = trainer.args
    # At epoch 2, modify the required loss by a factor of req_scheduler
    if trainer.epoch >= 2 and hyp.req_scheduler > 0:
        trainer.args.req_loss = hyp.req_loss * hyp.req_scheduler


def on_train_batch_end(trainer):
    if trainer.args.reinforcement_loss:
        if not hasattr(trainer, "batch_i"):
            trainer.batch_i = -1
        trainer.batch_i += 1
        if trainer.batch_i % 10 == 0:
            saveFile(trainer)


def saveFile(trainer):
    with open(f"{trainer.save_dir}/t_norm_usage_{trainer.epoch}.txt", 'w+') as t_norm_usage_file: 
        # if model is DDP-wrapped
        if isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel):
            criterion = trainer.model.module.criterion
        else:
            criterion = trainer.model.criterion

        t_norm_usage = criterion.t_norm_usage
        t_norm_percentage = {}
        for key, value in t_norm_usage.items():
            if sum(t_norm_usage.values()) > 0:
                t_norm_percentage[key] = value / sum(t_norm_usage.values())
            else:
                t_norm_percentage[key] = 0

        t_norm_values = criterion.t_norm_values

        # Ensure all printed dictionaries follow a user-defined T-norm order
        tnorm_order = [
            "product", "minimum", "lukasiewicz", "drastic", "hamacher_product",
            "nilpotent_minimum", "schweizer_sklar", "hamacher", "frank",
            "yager", "sugeno_weber", "aczel_alsina"
        ]
        usage_ordered = {k: t_norm_usage[k] for k in tnorm_order if k in t_norm_usage}
        percentage_ordered = {k: t_norm_percentage[k] for k in tnorm_order if k in t_norm_percentage}
        t_norm_values = {k: t_norm_values[k] for k in tnorm_order if k in t_norm_values}

        usage_json = json.dumps(usage_ordered, indent=2)
        percentage_json = json.dumps(percentage_ordered, indent=2)
        ucb_json = json.dumps(t_norm_values, indent=2)
        
        t_norm_usage_file.write(usage_json)
        t_norm_usage_file.write("\n")
        t_norm_usage_file.write(percentage_json)
        t_norm_usage_file.write("\n")
        t_norm_usage_file.write(ucb_json)


def main():
    args = getArgs()

    if args.dataset == "road-r":
        dataset = ROAD_R(args.dataset_path)
        val_list = []
        if args.task == 1:
            folder_name = "task1_yolo"
            val_list = "2014-11-14-16-34-33_stereo_centre_06,2014-06-26-09-53-12_stereo_centre_02,2015-02-13-09-16-26_stereo_centre_02".split(",")
        elif args.task == 2:
            folder_name = "task2_yolo"
            val_list = "2014-06-26-09-53-12_stereo_centre_02,2015-02-13-09-16-26_stereo_centre_02".split(",")
        elif args.task == 3:
            folder_name = "task3_yolo"
            val_list = "2014-06-26-09-53-12_stereo_centre_02".split(",")
        elif args.task == 0:
            folder_name = "debug_yolo"
        const_path = "../constraints/constraints.npy"
        if not dataset.checkExists(folder_name) or args.remake:
            dataset.generateYOLO(dataset.getLabels(args.task), folder_name, seed=args.seed, val_split=args.val_split, val_list=val_list)
    elif args.dataset == "road++":
        dataset = ROAD_PP(args.dataset_path)
        folder_name = f"task{args.task}_yolo_roadpp"
        const_path = "../constraints/constraints_roadpp.npy"
        if not dataset.checkExists(folder_name) or args.remake:
            dataset.generateYOLO(dataset.getLabels(args.task), folder_name, seed=args.seed, val_split=args.val_split)

    folder_args = []
    folder_args.append(f"task{args.task}")
    folder_args.append(f"e{args.max_epochs}")
    folder_args.append(f"{args.basemodel}")
    folder_args.append(f"reqloss{args.req_loss}")
    if args.reinforcement_loss:
        folder_args.append("rl")
        folder_args.append(f"beta{args.beta_rl}")
        folder_args.append(f"delta{args.delta_rl}")
        folder_args.append(f"mode{args.rl_mode}")
    elif args.req_loss != 0:
        folder_args.append(args.req_type)
    else:
        folder_args.append("vanilla")
    
    folder_args.append("rsched" + str(args.req_scheduler))
    folder_args = "_".join(folder_args)


    if False:
        video_list = dataset.getLabels(args.task)
        val_list = dataset.getValidation(video_list, seed=args.seed, val_split=args.val_split)
        print("Train split:")
        print(sorted(video_list))
        print("Val split:")
        print(sorted(val_list))
        import csv

        with open("train_split_roadwr.csv", "w", newline="") as train_file:
            writer = csv.writer(train_file)
            for item in sorted(video_list):
                writer.writerow([item])

        with open("test_split_roadwr.csv", "w", newline="") as test_file:
            writer = csv.writer(test_file)
            for item in sorted(val_list):
                writer.writerow([item])
        return

    try:
        if not args.no_augment:
            trainer = MOD_YOLOTrainer(overrides={"device": args.cuda, "project": f"../runs/nparam/{folder_args}", "data":f"../config/dataset_task{args.task}.yaml", "task":"detect", "model":f"../models/{args.basemodel}.pt",
                                                "optimizer":args.optimizer, "lr0": args.lr, "epochs": args.max_epochs, "close_mosaic": 0, "req_loss": args.req_loss, "req_type": args.req_type, "reinforcement_loss": args.reinforcement_loss, "workers": args.workers, "freeze": args.freeze, "batch": 24, "max_det": args.max_det, "amp": True, "cache": False, "lrf": args.lrf, "req_num_detect": args.req_num_detect, "req_scheduler": args.req_scheduler, "const_path": const_path, "val": False, "beta_rl": args.beta_rl, "delta_rl": args.delta_rl, "rl_mode": args.rl_mode})

    except Exception as e:
        from ultralytics.utils import DEFAULT_CFG_PATH
        shutil.copyfile("../config/default.yaml", DEFAULT_CFG_PATH)
        print(f"Error: {e}")
        print("Finished copying default config file")
        print("Please run the script again")
        exit()

    trainer.add_callback('on_train_epoch_end', on_train_epoch_end)
    trainer.add_callback('on_train_batch_end', on_train_batch_end)
    trainer.train()

    if args.reinforcement_loss:
        with open(f"{trainer.save_dir}/t_norm_usage.txt", 'w+') as t_norm_usage_file:
            if isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel):
                criterion = trainer.model.module.criterion
            else:
                criterion = trainer.model.criterion
            
            t_norm_usage_file.write(json.dumps(criterion.t_norm_usage))

if __name__ == "__main__":
    main()