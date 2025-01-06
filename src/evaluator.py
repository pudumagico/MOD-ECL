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
from yolo.trainer import MOD_YOLOTrainer
from yolo.validator import MOD_YOLODetectionValidator


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
    parser.add_argument("-task", "--task", type=int, default=1, choices=[0, 1, 2, 3, 4], help="Task number")
    parser.add_argument("--model", type=str, default="train", help="Model to use")
    parser.add_argument("-dataset", "--dataset", type=str, default="road-r", help="Dataset to use")

    parser.add_argument("-dataset_path", "--dataset_path", type=str, default="../../road-dataset", help="Path to dataset")
    parser.add_argument("-c-path", "--constraints-path", type=str, default="../constraints/maxsat_constraints.wcnf", help="Path to constraints")
    parser.add_argument("--conf", type=float, default=0.05, help="Confidence threshold")
    parser.add_argument("--tracker", type=str, default="../config/botsort.yaml", help="Tracker configuration")
    parser.add_argument("-s", "--save", action="store_true", help="Save output")
    parser.add_argument("-threshold", "--threshold", type=float, default=0.3, help="Threshold for maxsat")

    
    return parser.parse_args()

from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel

class CustomYOLO(YOLO):
    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": MOD_YOLODetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            }
        }

def main():
    args = getArgs()

    folder = f"../runs/{args.model}/"
    import yaml

    with open(f"{folder}args.yaml") as stream:
        try:
            out_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    model = CustomYOLO(f"{folder}weights/best.pt")

    predictor = MOD_Predictor()
    model.predictor = predictor
    model.track = types.MethodType(track, model)

    metrics = model.val(data=f"../config/dataset_task{args.task}.yaml", device="cuda:0", batch=out_yaml['batch'], conf=args.conf, max_det=300, plots=False)
    # metrics = model.val(data=f"../config/dataset_task0.yaml", device="cuda:0", batch=out_yaml['batch'], conf=args.conf, max_det=300, plots=False)
    print(metrics)

    print("\n\n==============================RESULTS==============================")
    # print(metrics.results_dict)
    for k, v in metrics.results_dict.items():
        print(f"{k:30s}: {v:.6f}")

    
    # test = model.track(os.path.join(args.dataset_path, "road_test/videos", video_name + '.mp4'), save=args.save, save_txt=False, save_conf=False, device=args.cuda, project="../runs/debug", line_width=3, stream=True, tracker=args.tracker, conf=args.conf, max_det=300)
    


if __name__ == "__main__":
    main()