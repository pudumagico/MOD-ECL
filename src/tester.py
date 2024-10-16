import argparse
import os
from ultralytics import YOLO

from yolo.predictor import MOD_Predictor

import numpy as np
import torch
from pathlib import Path
from typing import Union
import types

import pickle
from constraint_validator import MaxSAT_Validator


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
    parser.add_argument("-c", "--cuda", type=str, default="0", help="CUDA device")
    parser.add_argument("-task", "--task", type=int, default=1, choices=[0, 1, 2], help="Task number")
    parser.add_argument("--model", type=str, default="train", help="Model to use")
    parser.add_argument("-dataset", "--dataset", type=str, default="road-r", help="Dataset to use")

    parser.add_argument("-dataset_path", "--dataset_path", type=str, default="../../road-dataset", help="Path to dataset")
    parser.add_argument("-c-path", "--constraints-path", type=str, default="../constraints/maxsat_constraints.wcnf", help="Path to constraints")
    parser.add_argument("--conf", type=float, default=0.05, help="Confidence threshold")
    parser.add_argument("--tracker", type=str, default="../config/botsort.yaml", help="Tracker configuration")
    parser.add_argument("-s", "--save", action="store_true", help="Save output")
    parser.add_argument("-threshold", "--threshold", type=float, default=0.3, help="Threshold for maxsat")

    
    return parser.parse_args()


def main():
    args = getArgs()
    if args.dataset == "road-r":
        videos = "2014-06-26-09-31-18_stereo_centre_02,2015-02-03-08-45-10_stereo_centre_04,"\
                        "2014-12-10-18-10-50_stereo_centre_02,2015-02-06-13-57-16_stereo_centre_01".split(",")
    else:
        assert False, "Dataset not supported"

    model = YOLO(f"../runs/task{args.task}_yolo/{args.model}/weights/best.pt")

    predictor = MOD_Predictor()
    model.predictor = predictor
    model.track = types.MethodType(track, model)
    if args.task == 2:
        maxsat_v = MaxSAT_Validator(args.constraints_path)
    
    db_final = {}
    for video_name in videos:
        test = model.track(os.path.join(args.dataset_path, "road_test/videos", video_name + '.mp4'), save=args.save, save_txt=False, save_conf=False, device=args.cuda, project="../runs/debug", line_width=3, stream=True, tracker=args.tracker, conf=args.conf, max_det=300)
        #test = model.predict(os.path.join(args.dataset_path, "road_test/rgb-images/", video_name + '/04639.jpg'), save=args.save, save_txt=False, save_conf=False, device=args.cuda, project="../runs/debug", line_width=3, stream=False, conf=args.conf, max_det=300)
        #exit()
        db = {}
        for res_id, res in enumerate(test):
            if args.save:
                if res_id > 1000:
                    exit()
            frame_name = "{:05}".format(int(res_id+1)) + ".jpg"
            frame_db = []
            for bbox_id in range(len(res.boxes)):
                bbox_db = {}
                bbox_db['bbox'] = res.boxes.xyxy[bbox_id].tolist()
                if args.task == 1:
                    bbox_db['labels'] = res.boxes.full_conf[bbox_id].tolist()
                else:
                    bbox_db['labels'] = maxsat_v.correct_v1(res.boxes.full_conf[bbox_id].tolist(), threshold=args.threshold)
                
                frame_db.append(bbox_db)

            db[frame_name] = frame_db
        db_final[video_name] = db

        with open(f"../result_output/final_results_task{args.task}.pkl", 'wb') as outfile:
            pickle.dump(db_final, outfile)


if __name__ == "__main__":
    main()