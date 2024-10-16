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


class ROAD_R:

    def getLabels(self, task):
        if task == 1:
            return "2014-07-14-14-49-50_stereo_centre_01,2015-02-03-19-43-11_stereo_centre_04,2015-02-24-12-32-19_stereo_centre_04".split(",")
        elif task == 2:
            return "2014-06-25-16-45-34_stereo_centre_02,2014-07-14-14-49-50_stereo_centre_01," \
                "2014-07-14-15-42-55_stereo_centre_03,2014-08-08-13-15-11_stereo_centre_01,2014-08-11-10-59-18_stereo_centre_02," \
                "2014-11-25-09-18-32_stereo_centre_04,2014-11-18-13-20-12_stereo_centre_05,2014-11-21-16-07-03_stereo_centre_01," \
                "2014-12-09-13-21-02_stereo_centre_01,2015-02-03-08-45-10_stereo_centre_02,2015-02-03-19-43-11_stereo_centre_04," \
                "2015-02-06-13-57-16_stereo_centre_02,2015-02-13-09-16-26_stereo_centre_05,2015-02-24-12-32-19_stereo_centre_04," \
                "2015-03-03-11-31-36_stereo_centre_01,2014-11-14-16-34-33_stereo_centre_06".split(",")
        elif task == 3:
            return "2014-06-25-16-45-34_stereo_centre_02,2014-07-14-14-49-50_stereo_centre_01," \
                "2014-07-14-15-42-55_stereo_centre_03,2014-08-08-13-15-11_stereo_centre_01,2014-08-11-10-59-18_stereo_centre_02," \
                "2014-11-25-09-18-32_stereo_centre_04,2014-11-18-13-20-12_stereo_centre_05,2014-11-21-16-07-03_stereo_centre_01," \
                "2014-12-09-13-21-02_stereo_centre_01,2015-02-03-08-45-10_stereo_centre_02,2015-02-03-19-43-11_stereo_centre_04," \
                "2015-02-06-13-57-16_stereo_centre_02,2015-02-13-09-16-26_stereo_centre_05,2015-02-24-12-32-19_stereo_centre_04," \
                "2015-03-03-11-31-36_stereo_centre_01,2014-11-14-16-34-33_stereo_centre_06".split(",")
        elif task == 0:
            return ["2014-07-14-14-49-50_stereo_centre_01"]
        else:
            assert False, "Invalid task number"


    def __init__(self, path="../../road-dataset") -> None:
        self.path = path

    def checkExists(self, folder_name):
        return os.path.isdir(os.path.join(self.path, folder_name))

    def generateYOLO(self, video_list, folder_name, seed=1, val_split=0.2, val_list=[]):
        random.seed(seed)

        with open(os.path.join(self.path, "road/road_trainval_v1.0.json")) as f:
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
                    if (label in val_list) or (len(val_list) == 0 and random.random() < val_split):
                        mode = "val"
                        out_id = "{:05}".format(val_total)
                        val_total += 1
                    else:
                        mode = "train"
                        out_id = "{:05}".format(train_total)
                        train_total += 1
                    
                    frame_name = "{:05}".format(int(frame))
                    frame_now = label_info['db'][label]['frames'][frame]

                    shutil.copyfile(os.path.join(self.path, "road/rgb-images/", label, frame_name + ".jpg"), os.path.join(image_path, mode, out_id + ".jpg"))

                    total_output = []
                    if frame_now['annotated'] == 1:
                        for box_name in frame_now['annos'].keys():
                            id_labels = []
                            id_labels += filter_labels(frame_now['annos'][box_name]['agent_ids'], label_info['all_agent_labels'], label_info['agent_labels'], 0) 
                            id_labels += filter_labels(frame_now['annos'][box_name]['action_ids'], label_info['all_action_labels'], label_info['action_labels'], len(label_info['agent_labels']))
                            id_labels += filter_labels(frame_now['annos'][box_name]['loc_ids'], label_info['all_loc_labels'], label_info['loc_labels'], len(label_info['agent_labels']) + len(label_info['action_labels']))

                            box = ops.xyxy2xywh(np.asarray(frame_now['annos'][box_name]['box'])).tolist()
                            box = list(map(str, box))

                            total_output.append(','.join(list(map(str, id_labels))) + " " + ' '.join(box) + "\n")

                    with open(os.path.join(label_path, mode, out_id + ".txt"), "w") as f:
                        f.writelines(total_output)
            
