import argparse
import os
import torchmetrics.detection
import torchmetrics.detection.mean_ap
from ultralytics import YOLO

from yolo.predictor import MOD_Predictor

import numpy as np
import torch
from pathlib import Path
from typing import Union
import types

import pickle
from constraint_validator import MaxSAT_Validator
import ultralytics.utils.ops as ops
import ultralytics.utils.metrics as metrics
import torchmetrics


import csv


def track(
        prob_self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        persist: bool = False,
        **kwargs,
    ) -> list:
        """
        Conducts object tracking on the specified input source using the registered trackers.

        This method performs object tracking using the model's predictors and optionally registered trackers. It is
        capable of handling different types of input sources such as file paths or video streams. The method supports
        customization of the tracking process through various keyword arguments. It registers trackers if they are not
        already present and optionally persists them based on the 'persist' flag.

        The method sets a default confidence threshold specifically for ByteTrack-based tracking, which requires low
        confidence predictions as input. The tracking mode is explicitly set in the keyword arguments.

        Args:
            source (str, optional): The input source for object tracking. It can be a file path, URL, or video stream.
            stream (bool, optional): Treats the input source as a continuous video stream. Defaults to False.
            persist (bool, optional): Persists the trackers between different calls to this method. Defaults to False.
            **kwargs (any): Additional keyword arguments for configuring the tracking process. These arguments allow
                for further customization of the tracking behavior.

        Returns:
            (List[ultralytics.engine.results.Results]): A list of tracking results, encapsulated in the Results class.

        Raises:
            AttributeError: If the predictor does not have registered trackers.
        """
        if not hasattr(prob_self.predictor, "trackers"):
            from yolo.track import register_tracker

            register_tracker(prob_self, persist)
        kwargs["conf"] = kwargs.get("conf") or 0.1  # ByteTrack-based method needs low confidence predictions as input
        kwargs["batch"] = kwargs.get("batch") or 1  # batch-size 1 for tracking in videos
        kwargs["mode"] = "track"
        return prob_self.predict(source=source, stream=stream, **kwargs)


def getArgs():
    parser = argparse.ArgumentParser(description='MOD-CL')
    parser.add_argument("-c", "--cuda", type=str, default="cuda:0", help="CUDA device")
    parser.add_argument("-task", "--task", type=int, default=1, choices=[0, 1, 2, 3, 4], help="Task number")
    parser.add_argument("--model", type=str, default="train", help="Model to use")
    parser.add_argument("-dataset", "--dataset", type=str, default="road-r", help="Dataset to use")

    parser.add_argument("-dataset_path", "--dataset_path", type=str, default="../../road-dataset", help="Path to dataset")
    parser.add_argument("-c-path", "--constraints-path", type=str, default="../constraints/maxsat_constraints.wcnf", help="Path to constraints")
    parser.add_argument("--conf", type=float, default=0.05, help="Confidence threshold")
    parser.add_argument("--tracker", type=str, default="../config/botsort.yaml", help="Tracker configuration")
    parser.add_argument("-s", "--save", action="store_true", help="Save output")
    parser.add_argument("-threshold", "--threshold", type=float, default=0.3, help="Threshold for maxsat")

    parser.add_argument("-stats", "--stats", action="store_true", help="Print stats")
    parser.add_argument("-pred", "--prediction", action="store_true", help="Use prediction without tracking")
    parser.add_argument("-maxsat", "--maxsat", action="store_true", help="Use MaxSAT solvers to validate results")
    parser.add_argument("-iou", "--iou", type=float, default=0.7, help="IOU threshold for tracking")
    
    return parser.parse_args()

def calcViolation(full_conf, constraints, threshold):
        """Calculate the violation of the constraints."""
        pred_const = (full_conf >= threshold).float()

        pred_const = pred_const[pred_const.sum(-1) > 0]
        pred_const = torch.cat([pred_const, 1-pred_const], axis=-1) # Invert the values
        loss_const = torch.ones((pred_const.shape[0], constraints.shape[0]))
        for req_id in range(constraints.shape[0]):
            req_indices = constraints.indices()[1][constraints.indices()[0]==req_id]
            loss_const[:,req_id] = 1-torch.max(pred_const[:,req_indices], axis=-1)[0] # Violation of constraints
        full_violation = (loss_const).sum() / loss_const.shape[1]
        perbox_violation = (loss_const.sum(-1) > 0).float().sum()
    
        # temp = constraints.indices()[1][constraints.indices()[0]==0]
        # print(torch.argwhere(loss_const == 1))
        return float(full_violation), float(perbox_violation), int(full_conf.shape[0])

from torch_scatter import scatter_max
def calcViolation2(full_conf, constraints, threshold):
    """
    Calculate the violation of the constraints using torch-scatter.
    """
    # Thresholding to get binary predictions
    pred_const = (full_conf >= threshold).float()

    # Filter out boxes with no active constraints
    pred_const = pred_const[pred_const.sum(-1) > 0]

    # Add inverted values to create a two-channel representation
    pred_const = torch.cat([pred_const, 1 - pred_const], dim=-1)

    # Prepare the `loss_const` tensor
    num_constraints = constraints.shape[0]
    loss_const = torch.ones((pred_const.shape[0], num_constraints), device=pred_const.device)

    # Extract indices from constraints
    req_ids = constraints.indices()[0]  # Constraint IDs
    req_indices = constraints.indices()[1]  # Box indices for each constraint

    req_ids = req_ids.to(pred_const.device)
    req_indices = req_indices.to(pred_const.device)

    # Use scatter_max to compute max violations per constraint
    max_values, _ = scatter_max(
        pred_const[:, req_indices],  # Gather values from pred_const
        req_ids,                     # Group by constraint IDs
        dim=1,                       # Reduce along the constraints dimension
        dim_size=num_constraints     # Ensure the correct size
    )

    # Compute constraint violations
    loss_const = 1 - max_values
    
    # temp = constraints.indices()[1][constraints.indices()[0]==0]
    # print(torch.argwhere(loss_const == 1))

    # Compute full and per-box violations
    full_violation = loss_const.sum() / num_constraints
    perbox_violation = (loss_const.sum(dim=-1) > 0).float().sum()

    return float(full_violation), float(perbox_violation), int(full_conf.shape[0])

def main():
    args = getArgs()
    if args.dataset == "road-r":
        from dataset.road_r import ROAD_R
        # folder = "road_test/videos"
        # gt_folder = "../../road-dataset/road/road_trainval_v1.0.json"
        # videos = "2014-06-26-09-31-18_stereo_centre_02,2015-02-03-08-45-10_stereo_centre_04,"\
        #                 "2014-12-10-18-10-50_stereo_centre_02,2015-02-06-13-57-16_stereo_centre_01".split(",")
        
        folder = "road/videos"
        gt_folder = "../../road-dataset/road/road_trainval_v1.0.json"
        dataset = ROAD_R(args.dataset_path)

        if args.task == 1:
            videos = "2014-11-14-16-34-33_stereo_centre_06,2014-06-26-09-53-12_stereo_centre_02,2015-02-13-09-16-26_stereo_centre_02".split(",")
        elif args.task == 2:
            videos = "2014-06-26-09-53-12_stereo_centre_02,2015-02-13-09-16-26_stereo_centre_02".split(",")
        elif args.task == 3:
            videos = "2014-06-26-09-53-12_stereo_centre_02".split(",")
    elif args.dataset == 'road++r':
        from dataset.waymo_road import ROAD_PP
        folder = "train/videos"
        gt_folder = "../../ROAD++/road_waymo_trainval_v1.0.json"
        dataset = ROAD_PP(args.dataset_path)
        videos = dataset.getValidation(dataset.getLabels(args.task), seed=1, val_split=0.2)

        videos = ["train_00000"]
        gt_folder = "label_info.pkl"
    else:
        assert False, "Dataset not supported"

    model = YOLO(f"../runs/{args.model}/weights/best.pt")

    if args.stats:
        import json
        label_info = {}
        if gt_folder.endswith('.json'):
            with open(gt_folder) as f:
                label_info = json.load(f)
        else:
            with open(gt_folder, 'rb') as f:
                label_info = pickle.load(f)
        if args.dataset == 'road++r':
            constraints = torch.from_numpy(np.load("../constraints/constraints_roadpp.npy")).to_sparse()
        else:
            constraints = torch.from_numpy(np.load("../constraints/constraints.npy")).to_sparse()

    predictor = MOD_Predictor()
    predictor.dataset_type = args.dataset
    model.predictor = predictor
    model.track = types.MethodType(track, model)
    if args.task == 2 or args.maxsat:
        maxsat_v = MaxSAT_Validator(args.constraints_path)
    
    db_final = {}
    frame_map = []
    perbox_violation = []
    full_violation = []
    pred_box_num = []

    pred_frame_num = []

    mode = model.track if not args.prediction else model.predict

    for video_name in videos:
        test = mode(os.path.join(args.dataset_path, folder, video_name + '.mp4'), save=args.save, save_txt=False, save_conf=False, device=args.cuda, project="../runs/debug", line_width=3, stream=True, tracker=args.tracker, conf=args.conf, max_det=300, workers=8, iou=args.iou)
        # test = model.predict(os.path.join(args.dataset_path, "road_test/rgb-images/", video_name + '/04639.jpg'), save=args.save, save_txt=False, save_conf=False, device=args.cuda, project="../runs/debug", line_width=3, stream=False, conf=args.conf, max_det=300)
        # exit()
        db = {}
        perbox_violation_now = []
        full_violation_now = []
        pred_box_num_now = 0
        frame_map_now = []

        for res_id, res in enumerate(test):
            if args.save:
                if res_id > 1000:
                    exit()
            frame_name = "{:05}".format(int(res_id+1)) + ".jpg"
            frame_db = []



            if args.stats:
                gt_box = []
                gt_class = []
                h, w = res.orig_shape
                if str(res_id+1)in label_info['db'][video_name]['frames'].keys() and 'annos' in label_info['db'][video_name]['frames'][str(res_id+1)].keys() and len(label_info['db'][video_name]['frames'][str(res_id+1)]['annos']) > 0:
                    frame_now = label_info['db'][video_name]['frames'][str(res_id+1)]
                    for box_name in frame_now['annos'].keys():
                        box_now = frame_now['annos'][box_name]['box']
                        box_now[0] *= w
                        box_now[1] *= h
                        box_now[2] *= w
                        box_now[3] *= h
                        label_list = dataset.getLabelList(label_info, frame_now, box_name)
                        # for label in label_list:
                        for cl in label_list:
                            gt_box.append(torch.tensor(box_now, device=res.boxes.xyxy.device))
                            gt_class.append(torch.tensor(cl, device=res.boxes.xyxy.device))
                    gt = dict(
                        boxes=torch.stack(gt_box).to(res.boxes.xyxy.device),
                        labels=torch.stack(gt_class).to(res.boxes.xyxy.device),
                    )
                else:
                    gt = dict(
                        boxes=torch.tensor([], device=res.boxes.xyxy.device),
                        labels=torch.tensor([], device=res.boxes.xyxy.device),
                    )

                pred_box = []
                pred_class = []
                pred_conf = []

                if len(res.boxes) != 0:
                    for box in range(len(res.boxes)):
                        if args.maxsat:
                            for cl in maxsat_v.correct_v1(res.boxes.full_conf[box].tolist(), threshold=args.conf):
                                pred_box.append(res.boxes.xyxy[box])
                                pred_class.append(torch.tensor(cl, device=res.boxes.xyxy.device))
                                pred_conf.append(torch.tensor(1.0, device=res.boxes.xyxy.device))
                            full_violation_now.append(0)
                            perbox_violation_now.append(0)
                            pred_box_num_now += len(res.boxes)
                        else:
                            # violations = [0, 0, 0]
                            for cl in torch.argwhere(res.boxes.full_conf[box] > args.conf):
                                pred_box.append(res.boxes.xyxy[box])
                                pred_class.append(cl[0])
                                pred_conf.append(res.boxes.full_conf[box,cl[0]])
                                # violation_now = calcViolation(res.boxes.full_conf[box], constraints, args.conf)
                                # violations[0] += violation_now[0]
                                # violations[1] += violation_now[1]
                                # violations[2] += violation_now[2]
                            # violations = calcViolation(res.boxes.full_conf, constraints, args.conf)
                            violations = calcViolation2(res.boxes.full_conf, constraints, args.conf)
                            full_violation_now.append(violations[0])
                            perbox_violation_now.append(violations[1])
                            pred_box_num_now += violations[2]
                
                    pred = dict(
                        boxes=torch.stack(pred_box).to(res.boxes.xyxy.device),
                        labels=torch.stack(pred_class).to(res.boxes.xyxy.device),
                        scores=torch.stack(pred_conf).to(res.boxes.xyxy.device)
                    )


                else:
                    pred = dict(
                        boxes=torch.tensor([], device=res.boxes.xyxy.device),
                        labels=torch.tensor([], device=res.boxes.xyxy.device),
                        scores=torch.tensor([], device=res.boxes.xyxy.device)
                    )
                
                if len(gt['boxes']) != 0 or len(pred['boxes']) != 0:
                    metric = torchmetrics.detection.MeanAveragePrecision('xyxy')
                    out = float(metric.forward([pred], [gt])['map_50'])
                    if out < 0:
                        out = 0
                    frame_map_now.append(out)


            else:
                for bbox_id in range(len(res.boxes)):
                    bbox_db = {}
                    bbox_db['bbox'] = res.boxes.xyxy[bbox_id].tolist()

                    if args.task != 2:
                        bbox_db['labels'] = res.boxes.full_conf[bbox_id].tolist()
                    else:
                        bbox_db['labels'] = maxsat_v.correct_v1(res.boxes.full_conf[bbox_id].tolist(), threshold=args.threshold)
                    
                    frame_db.append(bbox_db)

                db[frame_name] = frame_db

        if args.stats:
            full_violation.append(np.sum(full_violation_now))
            perbox_violation.append(np.sum(perbox_violation_now))
            pred_box_num.append(pred_box_num_now)
            frame_map.append(np.sum(frame_map_now))
            pred_frame_num.append(len(frame_map_now))

            print(f"Video: {video_name}")
            print(f"Mean mAP: {np.sum(frame_map)/np.sum(pred_frame_num)}")
            print(f"Mean full violation: {np.sum(full_violation)/np.sum(pred_box_num)}")
            print(f"Mean perbox violation: {np.sum(perbox_violation/np.sum(pred_box_num))}")
            with open(f"../runs/{args.model}/stats_{'track' if not args.prediction else 'pred'}{'_maxsat' if args.maxsat else ''}.csv", 'w') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(["Video", "Mean mAP", "Mean full violation", "Mean perbox violation", "Total boxes", "Total frames", "Total violations"])
                for v, f, p, pb, pbn, pfn in zip(videos, frame_map, full_violation, perbox_violation, pred_box_num, pred_frame_num):
                    csv_writer.writerow([v, f/pfn, p/pbn, pb/pbn, pbn, pfn, pb])
                csv_writer.writerow(["Total", np.sum(frame_map)/np.sum(pred_frame_num), np.sum(full_violation)/np.sum(pred_box_num), np.sum(perbox_violation)/np.sum(pred_box_num), np.sum(pred_box_num), np.sum(pred_frame_num), np.sum(perbox_violation)])
        else:
            db_final[video_name] = db
            with open(f"../result_output/final_results_task{args.task}.pkl", 'wb') as outfile:
                pickle.dump(db_final, outfile)


if __name__ == "__main__":
    main()