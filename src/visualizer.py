"""
import pickle
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import os
import shutil
import torch


def checkViolation(constraints, pred, threshold):
    if len(pred.shape) == 1:
        pred = pred.unsqueeze(0)
    # round with threshold of threshold
    pred = (pred > threshold).float()
    pred_const = torch.cat([pred, 1-pred], axis=-1)
    loss_const = torch.zeros((pred_const.shape[0], constraints.shape[0]))
    for req_id in range(constraints.shape[0]):
        req_ind = constraints.indices()[1][constraints.indices()[0]==req_id]
        if len(req_ind) == 0:
            continue
        fuzzy_values = 1 - pred_const[:,req_ind]
        loss_const[:,req_id] = torch.min(fuzzy_values, axis=-1)[0]
    return int((loss_const.max(-1)[0]).sum() / (loss_const.shape[0]))


def overlaps(box, prev_boxes):
    for prev_box in prev_boxes:
        # Calculate intersection coordinates
        x1 = max(box[0], prev_box[0])
        y1 = max(box[1], prev_box[1])
        x2 = min(box[2], prev_box[2])
        y2 = min(box[3], prev_box[3])
        
        # Calculate intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate areas of each box
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_prev_box = (prev_box[2] - prev_box[0]) * (prev_box[3] - prev_box[1])
        
        # Calculate IoU
        union = area_box + area_prev_box - intersection
        if union == 0:  # Handle division by zero
            iou = 0
        else:
            iou = intersection / union
        
        # Return True if IoU exceeds threshold
        if iou > 0.7:
            return True
            
    return False


label_names = [
    "Ped",
    "Car",
    "Cyc",
    "Mobike",
    "MedVeh",
    "LarVeh",
    "Bus",
    "EmVeh",
    "TL",
    "OthTL",
    "Red",
    "Amber",
    "Green",
    "MovAway",
    "MovTow",
    "Mov",
    "Brake",
    "Stop",
    "IncatLeft",
    "IncatRht",
    "HazLit",
    "TurnLft",
    "TurnRht",
    "Ovtak",
    "Wait2X",
    "XingFmLft",
    "XingFmRht",
    "Xing",
    "PushObj",
    "VehLane",
    "OutgoLane",
    "OutgoCycLane",
    "IncomLane",
    "IncomCycLane",
    "Pav",
    "LftPav",
    "RhtPav",
    "Jun",
    "xing",
    "BusStop",
    "parking"
]

# import json
# # folder = "road-dataset/road/road_trainval_v1.0.json"
# folder = "ROAD++/road_waymo_trainval_v1.0.json"

# with open(os.path.join(folder)) as f:
#     label_info = json.load(f)


# color = (0, 0, 255)
constraints = torch.from_numpy(np.load("../MOD-CL-prev/YOLO/constraints.npy")).to_sparse()

processed = False
fps = 10
# temp = ["../MOD-CL-prev/result_output/final_output_task2.pkl", "../MOD-CL-prev/result_output/final_validated_output.pkl"]
# temp = ["../MOD-CL-prev/result_output/final_validated_output.pkl"]
temp = ["../MOD-CL-prev/result_output/final_output_task1.pkl", "../MOD-CL-prev/result_output/final_output_task1_stage2.pkl"]
videos = "2014-06-26-09-31-18_stereo_centre_02,2015-02-03-08-45-10_stereo_centre_04,"\
                    "2014-12-10-18-10-50_stereo_centre_02,2015-02-06-13-57-16_stereo_centre_01".split(",")
dataset_folder = "../road-dataset/road_test/rgb-images/"

temp = ["../T-NormRL/result_output/final_results_task4.pkl", "../T-NormRL/result_output/final_results_task4.pkl"]
videos = ["train_00000"]
dataset_folder = "../ROAD++/train/rgb-images/"

if os.path.exists("out"):
    shutil.rmtree("out")
os.makedirs("out/mistake")
mistake_set = 1

with open(f"{temp[1]}", "rb") as f:
    data_temp = pickle.load(f)

with open(f"{temp[0]}", "rb") as f:
    data = pickle.load(f)
    #for i in tqdm(range(1, 2)):
    for video in videos:
        os.makedirs(f"out/{video}")
        for i in tqdm(range(1, data[video].keys().__len__() + 1)):
            mistake = False
            if i % fps != 0:
                continue
            frame_name = '{:05d}.jpg'.format(i)
            #print(data['2014-06-26-09-31-18_stereo_centre_02'][frame_name][0])



            # Load the image
            image_path = f'{dataset_folder}{video}/{frame_name}'
            image = cv2.imread(image_path)

            # Bounding box data
            bbox_data = data[video][frame_name]

            # Draw bounding boxes and labels
            num_mistakes = 0
            num_boxes = 0
            previous_boxes = []
            prev_labels = []

            for entry in bbox_data:
                bbox = entry['bbox']
                labels = entry['labels']
                mistake_now = False
                

                # Extract coordinates
                x1, y1, x2, y2 = map(int, bbox)
                
                # Prepare label text
                if processed:
                    confident_labels = [label_names[idx] for idx in labels]
                else:
                    confident_labels = [label_names[idx] for idx, confidence in enumerate(labels) if confidence >= self.threshold]

                    if len(confident_labels) == 0:
                        #print("No confident labels found for this bounding box")
                        label_text = "No confident labels found"
                        continue
                    

                    mistake_now = checkViolation(constraints, torch.tensor(entry['labels']), threshold=self.threshold) >= 1

                    if not overlaps([x1, y1, x2, y2], previous_boxes):
                        num_mistakes += 1 if mistake_now else 0
                        num_boxes += 1
                        previous_boxes.append([x1, y1, x2, y2])
                        prev_labels.append(confident_labels)
                

                # Draw bounding box
                color = (0, 0, 255) if mistake_now else (0, 255, 0)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                

                # Put text on the image
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_thickness = 2
                line_height = cv2.getTextSize('Sample', font, font_scale, font_thickness)[0][1] + 5
                for i, line in enumerate(confident_labels):
                    y_position = y1 - 5 + (i - len(confident_labels) + 1) * line_height
                    cv2.putText(image, line, (x1, y_position), font, font_scale, color, font_thickness)
                
                if not mistake and mistake_now:
                    mistake = True


            # Alternatively, you can save the image with bounding boxes
            output_image_path = f'out/{video}/{frame_name}'
            cv2.imwrite(output_image_path, image)
            

            if mistake and num_mistakes / num_boxes >= 0.5 and num_boxes > 2:
                if False:
                    cv2.imwrite(f'out/mistake/{num_mistakes/num_boxes + num_boxes/100:.2f}_{videos.index(video)}_{frame_name}', image)
                else:
                    folder = f'out/mistake/{num_mistakes/num_boxes + num_boxes/100:.2f}_{mistake_set}'
                    os.makedirs(folder, exist_ok=True)
                    cv2.imwrite(f'{folder}/stage1_{frame_name}', image)
                    # Making post-processed version
                    image = cv2.imread(image_path)

                    # Bounding box data
                    bbox_data = data_temp[video][frame_name]

                    # Draw bounding boxes and labels
                    num_mistakes_new = 0
                    for entry in bbox_data:
                        bbox = entry['bbox']
                        labels = entry['labels']
                        

                        # Extract coordinates
                        x1, y1, x2, y2 = map(int, bbox)
                        
                        # Prepare label text
                        confident_labels = [label_names[idx] for idx, confidence in enumerate(labels) if confidence >= self.threshold]

                        if len(confident_labels) == 0:
                            #print("No confident labels found for this bounding box")
                            label_text = "No confident labels found"
                            continue

                        mistake_now = checkViolation(constraints, torch.tensor(entry['labels']), threshold=self.threshold) >= 1
                        num_mistakes_new += 1 if mistake_now else 0
                        
                        # Draw bounding box
                        color = (0, 255, 0) if not mistake_now else (0, 0, 255)
                        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                        

                        # Put text on the image
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1
                        font_thickness = 2
                        line_height = cv2.getTextSize('Sample', font, font_scale, font_thickness)[0][1] + 5
                        for i, line in enumerate(confident_labels):
                            y_position = y1 - 5 + (i - len(confident_labels) + 1) * line_height
                            cv2.putText(image, line, (x1, y_position), font, font_scale, color, font_thickness)
                        
                    
                    cv2.imwrite(f'{folder}/stage2_{frame_name}', image)
                    if num_mistakes_new != num_mistakes:
                        print(f"Stage 2 changed mistakes from {num_mistakes} to {num_mistakes_new} for {folder}") 
                    
                        shutil.copytree(f'{folder}', f'out/mistake2/{mistake_set}')
                        shutil.rmtree(folder)
                    mistake_set += 1

"""


# Make a better visualization class code based on the above snippet
import pickle
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import os
import shutil
import torch

class Visualizer:
    def __init__(self, constraints, dataset_folder, label_names, processed, output_folder="out", fps=10, threshold=0.3):
        self.constraints = constraints
        self.dataset_folder = dataset_folder
        self.output_folder = output_folder
        self.fps = fps
        self.label_names = label_names
        self.processed = processed
        self.mistake_set = 1
        self.threshold = threshold
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        else:
            shutil.rmtree(self.output_folder)
            os.makedirs(self.output_folder)

    def check_violation(self, constraints, pred, threshold):
        if len(pred.shape) == 1:
            pred = pred.unsqueeze(0)
        pred = (pred > threshold).float()
        pred_const = torch.cat([pred, 1 - pred], axis=-1)
        loss_const = torch.zeros((pred_const.shape[0], constraints.shape[0]))
        for req_id in range(constraints.shape[0]):
            req_ind = constraints.indices()[1][constraints.indices()[0] == req_id]
            if len(req_ind) == 0:
                continue
            fuzzy_values = 1 - pred_const[:, req_ind]
            loss_const[:, req_id] = torch.min(fuzzy_values, axis=-1)[0]
        return int((loss_const.max(-1)[0]).sum() / (loss_const.shape[0]))

    def overlaps(self, box, prev_boxes):
        for prev_box in prev_boxes:
            x1 = max(box[0], prev_box[0])
            y1 = max(box[1], prev_box[1])
            x2 = min(box[2], prev_box[2])
            y2 = min(box[3], prev_box[3])
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area_box = (box[2] - box[0]) * (box[3] - box[1])
            area_prev_box = (prev_box[2] - prev_box[0]) * (prev_box[3] - prev_box[1])
            union = area_box + area_prev_box - intersection
            if union == 0:
                iou = 0
            else:
                iou = intersection / union
            if iou > 0.7:
                return True
        return False

    def visualize(self, data_paths, videos):
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)
        os.makedirs(f"{self.output_folder}/mistake")

        with open(data_paths[1], "rb") as f:
            data_temp = pickle.load(f)

        with open(data_paths[0], "rb") as f:
            data = pickle.load(f)
            for video in videos:
                os.makedirs(f"{self.output_folder}/{video}")
                for i in tqdm(range(1, len(data[video]) + 1)):
                    if i % self.fps != 0:
                        continue
                    frame_name = '{:05d}.jpg'.format(i)
                    image_path = f'{self.dataset_folder}{video}/{frame_name}'
                    image = cv2.imread(image_path)
                    bbox_data = data[video][frame_name]
                    self.process_frame(image, bbox_data, frame_name, video, data_temp)

    def process_frame(self, image, bbox_data, frame_name, video, data_temp):
        num_mistakes = 0
        num_boxes = 0
        previous_boxes = []
        mistake = False

        for entry in bbox_data:
            bbox = entry['bbox']
            labels = entry['labels']
            mistake_now = False
            x1, y1, x2, y2 = map(int, bbox)
            confident_labels = self.get_confident_labels(labels)

            if not confident_labels:
                continue

            mistake_now = self.check_violation(self.constraints, torch.tensor(entry['labels']), threshold=self.threshold) >= 1

            if not self.overlaps([x1, y1, x2, y2], previous_boxes):
                num_mistakes += 1 if mistake_now else 0
                num_boxes += 1
                previous_boxes.append([x1, y1, x2, y2])

            color = (0, 0, 255) if mistake_now else (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            self.put_labels_on_image(image, confident_labels, x1, y1, color)

            if not mistake and mistake_now:
                mistake = True

        output_image_path = f'{self.output_folder}/{video}/{frame_name}'
        if not os.path.exists(f'{self.output_folder}/{video}'):
            os.makedirs(f'{self.output_folder}/{video}')
        cv2.imwrite(output_image_path, image)

        if mistake and num_mistakes / num_boxes >= 0.5 and num_boxes > 2:
            try:
                self.save_mistake_image(image, frame_name, video, num_mistakes, num_boxes, data_temp)
            except Exception as e:
                print(f"Error in saving mistake image for {frame_name} in {video}: {e}")

    def get_confident_labels(self, labels):
        if self.processed:
            return [self.label_names[idx] for idx in labels]
        else:
            return [self.label_names[idx] for idx, confidence in enumerate(labels) if confidence >= self.threshold]

    def put_labels_on_image(self, image, labels, x1, y1, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        line_height = cv2.getTextSize('Sample', font, font_scale, font_thickness)[0][1] + 5
        for i, line in enumerate(labels):
            y_position = y1 - 5 + (i - len(labels) + 1) * line_height
            cv2.putText(image, line, (x1, y_position), font, font_scale, color, font_thickness)

    def save_mistake_image(self, image, frame_name, video, num_mistakes, num_boxes, data_temp):
        folder = f'{self.output_folder}/mistake/{num_mistakes / num_boxes + num_boxes / 100:.2f}_{self.mistake_set}'
        os.makedirs(folder, exist_ok=True)
        cv2.imwrite(f'{folder}/stage1_{frame_name}', image)
        image = cv2.imread(f'{self.dataset_folder}{video}/{frame_name}')
        num_mistakes_new = 0

        for entry in data_temp:
            bbox = entry['bbox']
            labels = entry['labels']
            x1, y1, x2, y2 = map(int, bbox)
            confident_labels = self.get_confident_labels(labels)
            if not confident_labels:
                continue
            mistake_now = self.check_violation(self.constraints, torch.tensor(entry['labels']), threshold=self.threshold) >= 1
            num_mistakes_new += 1 if mistake_now else 0
            color = (0, 255, 0) if not mistake_now else (0, 0, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            self.put_labels_on_image(image, confident_labels, x1, y1, color)

        cv2.imwrite(f'{folder}/stage2_{frame_name}', image)
        if num_mistakes_new != num_mistakes:
            print(f"Stage 2 changed mistakes from {num_mistakes} to {num_mistakes_new} for {folder}")
            shutil.copytree(folder, f'{self.output_folder}/mistake2/{self.mistake_set}')
            shutil.rmtree(folder)
        self.mistake_set += 1

# Usage example
# visualizer = Visualizer(
#     constraints_path="../MOD-CL-prev/YOLO/constraints.npy",
#     dataset_folder="../ROAD++/train/rgb-images/"
# )
# visualizer.visualize(
#     data_paths=["../T-NormRL/result_output/final_results_task4.pkl", "../T-NormRL/result_output/final_results_task4.pkl"],
#     videos=["train_00000"]
# )