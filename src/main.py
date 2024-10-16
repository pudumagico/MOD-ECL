import argparse
import os
import shutil
import json 

from yolo.trainer import MOD_YOLOTrainer
#from ultralytics import 
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
    parser.add_argument("-workers", "--workers", type=int, default=0, help="Number of workers")

    parser.add_argument("-optimizer", "--optimizer", type=str, default="Adam", help="Optimizer to use")
    parser.add_argument("-lr", "--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("-vanilla", "--vanilla", action="store_true", help="Use vanilla YOLO")
    parser.add_argument("-freeze", "--freeze", type=int, default=0, help="Freeze layers")
    parser.add_argument("-req-type", "--req-type", type=str, default="product", help="Requirements Loss type")
    parser.add_argument("-no_augment", "--no_augment", action="store_true", help="Use Augmentation")
    parser.add_argument("-max_det", "--max_det", type=int, default=300, help="Maximum detections")

    parser.add_argument("-rl", "--reinforcement-loss", type=bool, default=False, help="Use Reinforcement Learning Loss")

    return parser.parse_args()

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
        
        if not dataset.checkExists(folder_name) or args.remake:
            dataset.generateYOLO(dataset.getLabels(args.task), folder_name, seed=args.seed, val_split=args.val_split, val_list=val_list)
    elif args.dataset == "road++":
        dataset = ROAD_PP(args.dataset_path)
        folder_name = f"task{args.task}_yolo_roadpp"

        if not dataset.checkExists(folder_name) or args.remake:
            dataset.generateYOLO(dataset.getLabels(args.task), folder_name, seed=args.seed, val_split=args.val_split)


    #if args.vanilla:

    try:
        if not args.no_augment:
            trainer = MOD_YOLOTrainer(overrides={"device":args.cuda, "project": f"../runs/{folder_name}", "data":f"../config/dataset_task{args.task}.yaml", "task":"detect", "model":f"../models/{args.basemodel}.pt",
                                                "optimizer":args.optimizer, "lr0": args.lr, "epochs": args.max_epochs, "close_mosaic": 0, "req_loss": args.req_loss, "req_type": args.req_type, "reinforcement_loss": args.reinforcement_loss, "workers": args.workers, "freeze": args.freeze, "batch": 24, "max_det": args.max_det, "amp": False})
        else:
            trainer = MOD_YOLOTrainer(overrides={"device":args.cuda, "project": f"../runs/{folder_name}", "data":f"../config/dataset_task{args.task}.yaml", "task":"detect", "model":f"../models/{args.basemodel}.pt",
                                                "optimizer":args.optimizer, "lr0": args.lr, "epochs": args.max_epochs, "close_mosaic": 0, "req_loss": args.req_loss, "req_type": args.req_type, "reinforcement_loss": args.reinforcement_loss, "workers": args.workers, "freeze": args.freeze, "batch": 32, "hsv_h": 0, "hsv_s": 0, "hsv_v": 0, "translate": 0, "scale": 0, "fliplr": 0, "mosaic": 0, "erasing": 0, "crop_fraction": 0})
    except Exception as e:
        from ultralytics.utils import DEFAULT_CFG_PATH
        shutil.copyfile("../config/default.yaml", DEFAULT_CFG_PATH)
        print(f"Error: {e}")
        print("Finished copying default config file")
        print("Please run the script again")
        exit()

    trainer.train()

    if args.reinforcement_loss:
        with open(f"{trainer.save_dir}/t_norm_usage.txt", 'w+') as t_norm_usage_file:
            t_norm_usage_file.write(json.dumps(trainer.model.criterion.t_norm_usage))




if __name__ == "__main__":
    main()