import os
import json
import random
import shutil

from ultralytics.utils import ops

import numpy as np
from tqdm import tqdm


def filter_labels(ids, all_labels, used_labels, offset):
    """Filter the used ids"""
    used_ids = []
    for id in ids:
        label = all_labels[id]
        if label in used_labels:
            used_ids.append(used_labels.index(label) + offset)
    
    return used_ids


class ROAD_PP:

    agent_labels = ['Ped', 'Car', 'Cyc', 'Mobike', 'MedVeh', 'LarVeh', 'Bus', 'EmVeh', 'TL']
    action_labels = ['Red', 'Amber', 'Green', 'MovAway', 'MovTow', 'Mov', 'Brake', 'Stop', 'IncatLft', 'IncatRht', 'HazLit', 'TurLft', 'TurRht', 'Ovtak', 'Wait2X', 'XingFmLft', 'XingFmRht', 'Xing', 'PushObj']
    loc_labels = ['VehLane', 'OutgoLane', 'OutgoCycLane', 'IncomLane', 'IncomCycLane', 'Pav', 'LftPav', 'RhtPav', 'Jun', 'xing', 'BusStop', 'parking', 'OutgoBusLane', 'IncomBusLane', 'OutgoBusLane']
    plus_labels = ['OutgoBusLane', 'IncomBusLane', 'rightParking', 'LftParking', 'Rev', 'SmalVeh', 'MovLft', 'MovRht']

    def getLabels(self, task):
        # Train: 798, Test: 202
        return [f"train_{i:05}" for i in range(798)]

    def getValidation(self, video_list, seed=1, val_split=0.2):
        val_list = []
        random.seed(seed)
        for video in list(video_list):
            if random.random() < val_split:
                val_list.append(video)
        return val_list

    def __init__(self, path="../../ROAD++") -> None:
        self.path = path
        self.all_labels = self.agent_labels + self.action_labels + self.loc_labels + self.plus_labels

    def checkExists(self, folder_name):
        return os.path.isdir(os.path.join(self.path, folder_name))

    def getLabelList(self, label_info, frame_now, box_name):
        id_labels = []
        id_labels_name = []
        for id in frame_now['annos'][box_name]['agent_ids']:
            id_labels_name.append(label_info['all_agent_labels'][id])
        for id in frame_now['annos'][box_name]['action_ids']:
            id_labels_name.append(label_info['all_action_labels'][id])
        for id in frame_now['annos'][box_name]['loc_ids']:
            id_labels_name.append(label_info['all_loc_labels'][id])
            
        for label in id_labels_name:
            if label in self.all_labels:
                id_labels.append(self.all_labels.index(label))
            else:
                raise ValueError(f"Label {label} not found in all_labels")

        # id_labels += filter_labels(frame_now['annos'][box_name]['agent_ids'], label_info['all_agent_labels'], self.agent_labels, 0)
        # id_labels += filter_labels(frame_now['annos'][box_name]['action_ids'], label_info['all_action_labels'], self.action_labels, len(self.agent_labels))
        # id_labels += filter_labels(frame_now['annos'][box_name]['loc_ids'], label_info['all_loc_labels'], self.loc_labels, len(self.agent_labels) + len(self.action_labels))
        return id_labels

    def generateYOLO(self, video_list, folder_name, seed=1, val_split=0.2, val_list=[]):
        random.seed(seed)
        if val_list == []:
            for video in list(video_list):
                if random.random() < val_split:
                    val_list.append(video)
                    video_list.remove(video)

        with open(os.path.join(self.path, "road_waymo_trainval_v1.0.json")) as f:
            label_info = json.load(f)

        image_path = os.path.join(self.path, folder_name, "images")
        label_path = os.path.join(self.path, folder_name, "labels")

        if os.path.isdir(image_path):
            shutil.rmtree(os.path.join(self.path, folder_name))

        os.makedirs(os.path.join(image_path, "train"))
        os.makedirs(os.path.join(image_path, "val"))
        os.makedirs(os.path.join(label_path, "train"))
        os.makedirs(os.path.join(label_path, "val"))
        
        train_total = 0
        val_total = 0
        
        for label in label_info['db'].keys():
            if label in video_list or label in val_list:
                print("Processing video: ", label)
                for frame in tqdm(label_info['db'][label]['frames']):
                    if label in val_list:
                        mode = "val"
                        out_id = "{:05}".format(val_total)
                        val_total += 1
                    else:
                        mode = "train"
                        out_id = "{:05}".format(train_total)
                        train_total += 1
                    
                    frame_name = "{:05}".format(int(frame))
                    frame_now = label_info['db'][label]['frames'][frame]

                    shutil.copyfile(os.path.join(self.path, "train/rgb-images/", label, frame_name + ".jpg"), os.path.join(image_path, mode, out_id + ".jpg"))

                    total_output = []
                    if 'annos' in frame_now.keys():
                        for box_name in frame_now['annos'].keys():
                            id_labels = []
                            # id_labels += filter_labels(frame_now['annos'][box_name]['agent_ids'], label_info['all_agent_labels'], label_info['agent_labels'], 0) 
                            # id_labels += filter_labels(frame_now['annos'][box_name]['action_ids'], label_info['all_action_labels'], label_info['action_labels'], len(label_info['agent_labels']))
                            # id_labels += filter_labels(frame_now['annos'][box_name]['loc_ids'], label_info['all_loc_labels'], label_info['loc_labels'], len(label_info['agent_labels']) + len(label_info['action_labels']))
                            id_labels += filter_labels(frame_now['annos'][box_name]['agent_ids'], label_info['all_agent_labels'], self.agent_labels, 0)
                            id_labels += filter_labels(frame_now['annos'][box_name]['action_ids'], label_info['all_action_labels'], self.action_labels, len(self.agent_labels))
                            id_labels += filter_labels(frame_now['annos'][box_name]['loc_ids'], label_info['all_loc_labels'], self.loc_labels, len(self.agent_labels) + len(self.action_labels))

                            box = ops.xyxy2xywh(np.asarray(frame_now['annos'][box_name]['box'])).tolist()
                            for i in range(4):
                                box[i] = min(max(0, box[i]), 1)
                            box = list(map(str, box))

                            total_output.append(','.join(list(map(str, id_labels))) + " " + ' '.join(box) + "\n")

                    with open(os.path.join(label_path, mode, out_id + ".txt"), "w") as f:
                        f.writelines(total_output)
            
