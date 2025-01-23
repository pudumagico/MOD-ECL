# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics Results, Boxes and Masks classes for handling inference results.

Usage: See https://docs.ultralytics.com/modes/predict/
"""

from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from ultralytics.data.augment import LetterBox
from ultralytics.utils import LOGGER, SimpleClass, ops
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.engine.results import Masks, Probs, Keypoints, OBB, BaseTensor


class MOD_Results(SimpleClass):
    """
    A class for storing and manipulating inference results.

    Attributes:
        orig_img (numpy.ndarray): Original image as a numpy array.
        orig_shape (tuple): Original image shape in (height, width) format.
        boxes (Boxes, optional): Object containing detection bounding boxes.
        masks (Masks, optional): Object containing detection masks.
        probs (Probs, optional): Object containing class probabilities for classification tasks.
        keypoints (Keypoints, optional): Object containing detected keypoints for each object.
        speed (dict): Dictionary of preprocess, inference, and postprocess speeds (ms/image).
        names (dict): Dictionary of class names.
        path (str): Path to the image file.

    Methods:
        update(boxes=None, masks=None, probs=None, obb=None): Updates object attributes with new detection results.
        cpu(): Returns a copy of the Results object with all tensors on CPU memory.
        numpy(): Returns a copy of the Results object with all tensors as numpy arrays.
        cuda(): Returns a copy of the Results object with all tensors on GPU memory.
        to(*args, **kwargs): Returns a copy of the Results object with tensors on a specified device and dtype.
        new(): Returns a new Results object with the same image, path, and names.
        plot(...): Plots detection results on an input image, returning an annotated image.
        show(): Show annotated results to screen.
        save(filename): Save annotated results to file.
        verbose(): Returns a log string for each task, detailing detections and classifications.
        save_txt(txt_file, save_conf=False): Saves detection results to a text file.
        save_crop(save_dir, file_name=Path("im.jpg")): Saves cropped detection images.
        tojson(normalize=False): Converts detection results to JSON format.
    """

    def __init__(self, orig_img, path, names, nc, boxes=None, masks=None, probs=None, keypoints=None, obb=None, thres=0.1) -> None:
        """
        Initialize the Results class.

        Args:
            orig_img (numpy.ndarray): The original image as a numpy array.
            path (str): The path to the image file.
            names (dict): A dictionary of class names.
            boxes (torch.tensor, optional): A 2D tensor of bounding box coordinates for each detection.
            masks (torch.tensor, optional): A 3D tensor of detection masks, where each mask is a binary image.
            probs (torch.tensor, optional): A 1D tensor of probabilities of each class for classification task.
            keypoints (torch.tensor, optional): A 2D tensor of keypoint coordinates for each detection.
            obb (torch.tensor, optional): A 2D tensor of oriented bounding box coordinates for each detection.
        """
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.boxes = MOD_Boxes(boxes, self.orig_shape, nc) if boxes is not None else None  # native size boxes
        self.masks = Masks(masks, self.orig_shape) if masks is not None else None  # native size or imgsz masks
        self.probs = Probs(probs) if probs is not None else None
        self.keypoints = Keypoints(keypoints, self.orig_shape) if keypoints is not None else None
        self.obb = OBB(obb, self.orig_shape) if obb is not None else None
        self.speed = {"preprocess": None, "inference": None, "postprocess": None}  # milliseconds per image
        self.names = names
        self.path = path
        self.save_dir = None
        self.thres = thres
        self._keys = "boxes", "masks", "probs", "keypoints", "obb"
        self.nc = nc

    def __getitem__(self, idx):
        """Return a Results object for the specified index."""
        return self._apply("__getitem__", idx)

    def __len__(self):
        """Return the number of detections in the Results object."""
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                return len(v)

    def update(self, boxes=None, masks=None, probs=None, obb=None):
        """Update the boxes, masks, and probs attributes of the Results object."""
        if boxes is not None:
            new_data = ops.clip_boxes(boxes, self.orig_shape)
            full_conf = self.boxes.full_conf.to(new_data.device)
            self.boxes = MOD_Boxes(torch.cat((new_data[:, :4], full_conf, new_data[:, 4:]), 1), self.orig_shape, self.nc)
        if masks is not None:
            self.masks = Masks(masks, self.orig_shape)
        if probs is not None:
            self.probs = probs
        if obb is not None:
            self.obb = OBB(obb, self.orig_shape)

    def _apply(self, fn, *args, **kwargs):
        """
        Applies a function to all non-empty attributes and returns a new Results object with modified attributes. This
        function is internally called by methods like .to(), .cuda(), .cpu(), etc.

        Args:
            fn (str): The name of the function to apply.
            *args: Variable length argument list to pass to the function.
            **kwargs: Arbitrary keyword arguments to pass to the function.

        Returns:
            Results: A new Results object with attributes modified by the applied function.
        """
        r = self.new()
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                setattr(r, k, getattr(v, fn)(*args, **kwargs))
        return r

    def cpu(self):
        """Return a copy of the Results object with all tensors on CPU memory."""
        return self._apply("cpu")

    def numpy(self):
        """Return a copy of the Results object with all tensors as numpy arrays."""
        return self._apply("numpy")

    def cuda(self):
        """Return a copy of the Results object with all tensors on GPU memory."""
        return self._apply("cuda")

    def to(self, *args, **kwargs):
        """Return a copy of the Results object with tensors on the specified device and dtype."""
        return self._apply("to", *args, **kwargs)

    def new(self):
        """Return a new Results object with the same image, path, and names."""
        return MOD_Results(orig_img=self.orig_img, path=self.path, names=self.names, nc=self.nc)

    def plot(
        self,
        conf=True,
        line_width=None,
        font_size=None,
        font="Arial.ttf",
        pil=False,
        img=None,
        im_gpu=None,
        kpt_radius=5,
        kpt_line=True,
        labels=True,
        boxes=True,
        masks=True,
        probs=True,
        show=False,
        save=False,
        filename=None,
    ):
        """
        Plots the detection results on an input RGB image. Accepts a numpy array (cv2) or a PIL Image.

        Args:
            conf (bool): Whether to plot the detection confidence score.
            line_width (float, optional): The line width of the bounding boxes. If None, it is scaled to the image size.
            font_size (float, optional): The font size of the text. If None, it is scaled to the image size.
            font (str): The font to use for the text.
            pil (bool): Whether to return the image as a PIL Image.
            img (numpy.ndarray): Plot to another image. if not, plot to original image.
            im_gpu (torch.Tensor): Normalized image in gpu with shape (1, 3, 640, 640), for faster mask plotting.
            kpt_radius (int, optional): Radius of the drawn keypoints. Default is 5.
            kpt_line (bool): Whether to draw lines connecting keypoints.
            labels (bool): Whether to plot the label of bounding boxes.
            boxes (bool): Whether to plot the bounding boxes.
            masks (bool): Whether to plot the masks.
            probs (bool): Whether to plot classification probability
            show (bool): Whether to display the annotated image directly.
            save (bool): Whether to save the annotated image to `filename`.
            filename (str): Filename to save image to if save is True.

        Returns:
            (numpy.ndarray): A numpy array of the annotated image.

        Example:
            ```python
            from PIL import Image
            from ultralytics import YOLO

            model = YOLO('yolov8n.pt')
            results = model('bus.jpg')  # results list
            for r in results:
                im_array = r.plot()  # plot a BGR numpy array of predictions
                im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                im.show()  # show image
                im.save('results.jpg')  # save image
            ```
        """
        if img is None and isinstance(self.orig_img, torch.Tensor):
            img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).to(torch.uint8).cpu().numpy()

        names = self.names
        is_obb = self.obb is not None
        pred_boxes, show_boxes = self.obb if is_obb else self.boxes, boxes
        pred_masks, show_masks = self.masks, masks
        pred_probs, show_probs = self.probs, probs
        annotator = Annotator(
            deepcopy(self.orig_img if img is None else img),
            line_width,
            font_size,
            font,
            pil or (pred_probs is not None and show_probs),  # Classify tasks default to pil=True
            example=names,
        )

        # Plot Segment results
        if pred_masks and show_masks:
            if im_gpu is None:
                img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
                im_gpu = (
                    torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device)
                    .permute(2, 0, 1)
                    .flip(0)
                    .contiguous()
                    / 255
                )
            idx = pred_boxes.cls if pred_boxes else range(len(pred_masks))
            annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu)

        # Plot Detect results
        if pred_boxes is not None and show_boxes:
            for d in reversed(pred_boxes):
                c, conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
                pos_c = torch.argwhere(d.full_conf[0] > self.thres).squeeze(1).tolist()
                if d.full_conf.shape[0] > 1:
                    print("ERROR")
                    exit()
                pos_names = [names[i] for i in pos_c]
                name = ("" if id is None else f"id:{id} ") + ",".join(pos_names)
                label = (f"{name} {conf:.2f}" if conf else name) if labels else None
                box = d.xyxyxyxy.reshape(-1, 4, 2).squeeze() if is_obb else d.xyxy.squeeze()
                annotator.box_label(box, label, color=colors(c, True), rotated=is_obb)

        # Plot Classify results
        if pred_probs is not None and show_probs:
            text = ",\n".join(f"{names[j] if names else j} {pred_probs.data[j]:.2f}" for j in pred_probs.top5)
            x = round(self.orig_shape[0] * 0.03)
            annotator.text([x, x], text, txt_color=(255, 255, 255))  # TODO: allow setting colors

        # Plot Pose results
        if self.keypoints is not None:
            for k in reversed(self.keypoints.data):
                annotator.kpts(k, self.orig_shape, radius=kpt_radius, kpt_line=kpt_line)

        # Show results
        if show:
            annotator.show(self.path)

        # Save results
        if save:
            annotator.save(filename)

        return annotator.result()

    def show(self, *args, **kwargs):
        """Show annotated results image."""
        self.plot(show=True, *args, **kwargs)

    def save(self, filename=None, *args, **kwargs):
        """Save annotated results image."""
        if not filename:
            filename = f"results_{Path(self.path).name}"
        self.plot(save=True, filename=filename, *args, **kwargs)
        return filename

    def verbose(self):
        """Return log string for each task."""
        log_string = ""
        probs = self.probs
        boxes = self.boxes
        if len(self) == 0:
            return log_string if probs is not None else f"{log_string}(no detections), "
        if probs is not None:
            log_string += f"{', '.join(f'{self.names[j]} {probs.data[j]:.2f}' for j in probs.top5)}, "
        if boxes:
            for c in boxes.cls.unique():
                n = (boxes.cls == c).sum()  # detections per class
                log_string += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
        return log_string

    def save_txt(self, txt_file, save_conf=False):
        """
        Save predictions into txt file.

        Args:
            txt_file (str): txt file path.
            save_conf (bool): save confidence score or not.
        """
        is_obb = self.obb is not None
        boxes = self.obb if is_obb else self.boxes
        masks = self.masks
        probs = self.probs
        kpts = self.keypoints
        texts = []
        if probs is not None:
            # Classify
            [texts.append(f"{probs.data[j]:.2f} {self.names[j]}") for j in probs.top5]
        elif boxes:
            # Detect/segment/pose
            for j, d in enumerate(boxes):
                c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
                line = (c, *(d.xyxyxyxyn.view(-1) if is_obb else d.xywhn.view(-1)))
                if masks:
                    seg = masks[j].xyn[0].copy().reshape(-1)  # reversed mask.xyn, (n,2) to (n*2)
                    line = (c, *seg)
                if kpts is not None:
                    kpt = torch.cat((kpts[j].xyn, kpts[j].conf[..., None]), 2) if kpts[j].has_visible else kpts[j].xyn
                    line += (*kpt.reshape(-1).tolist(),)
                line += (conf,) * save_conf + (() if id is None else (id,))
                texts.append(("%g " * len(line)).rstrip() % line)

        if texts:
            Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # make directory
            with open(txt_file, "a") as f:
                f.writelines(text + "\n" for text in texts)

    def save_crop(self, save_dir, file_name=Path("im.jpg")):
        """
        Save cropped predictions to `save_dir/cls/file_name.jpg`.

        Args:
            save_dir (str | pathlib.Path): Save path.
            file_name (str | pathlib.Path): File name.
        """
        if self.probs is not None:
            LOGGER.warning("WARNING ⚠️ Classify task do not support `save_crop`.")
            return
        if self.obb is not None:
            LOGGER.warning("WARNING ⚠️ OBB task do not support `save_crop`.")
            return
        for d in self.boxes:
            save_one_box(
                d.xyxy,
                self.orig_img.copy(),
                file=Path(save_dir) / self.names[int(d.cls)] / f"{Path(file_name)}.jpg",
                BGR=True,
            )

    def summary(self, normalize=False, decimals=5):
        """Convert the results to a summarized format."""
        if self.probs is not None:
            LOGGER.warning("Warning: Classify results do not support the `summary()` method yet.")
            return

        # Create list of detection dictionaries
        results = []
        data = self.boxes.data.cpu().tolist()
        h, w = self.orig_shape if normalize else (1, 1)
        for i, row in enumerate(data):  # xyxy, track_id if tracking, conf, class_id
            box = {
                "x1": round(row[0] / w, decimals),
                "y1": round(row[1] / h, decimals),
                "x2": round(row[2] / w, decimals),
                "y2": round(row[3] / h, decimals),
            }
            conf = round(row[-2], decimals)
            class_id = int(row[-1])
            result = {"name": self.names[class_id], "class": class_id, "confidence": conf, "box": box}
            if self.boxes.is_track:
                result["track_id"] = int(row[-3])  # track ID
            if self.masks:
                result["segments"] = {
                    "x": (self.masks.xy[i][:, 0] / w).round(decimals).tolist(),
                    "y": (self.masks.xy[i][:, 1] / h).round(decimals).tolist(),
                }
            if self.keypoints is not None:
                x, y, visible = self.keypoints[i].data[0].cpu().unbind(dim=1)  # torch Tensor
                result["keypoints"] = {
                    "x": (x / w).numpy().round(decimals).tolist(),  # decimals named argument required
                    "y": (y / h).numpy().round(decimals).tolist(),
                    "visible": visible.numpy().round(decimals).tolist(),
                }
            results.append(result)

        return results

    def tojson(self, normalize=False, decimals=5):
        """Convert the results to JSON format."""
        import json

        return json.dumps(self.summary(normalize=normalize, decimals=decimals), indent=2)



class MOD_Boxes(BaseTensor):
    """
    Manages detection boxes, providing easy access and manipulation of box coordinates, confidence scores, class
    identifiers, and optional tracking IDs. Supports multiple formats for box coordinates, including both absolute and
    normalized forms.

    Attributes:
        data (torch.Tensor): The raw tensor containing detection boxes and their associated data.
        orig_shape (tuple): The original image size as a tuple (height, width), used for normalization.
        is_track (bool): Indicates whether tracking IDs are included in the box data.

    Properties:
        xyxy (torch.Tensor | numpy.ndarray): Boxes in [x1, y1, x2, y2] format.
        conf (torch.Tensor | numpy.ndarray): Confidence scores for each box.
        cls (torch.Tensor | numpy.ndarray): Class labels for each box.
        id (torch.Tensor | numpy.ndarray, optional): Tracking IDs for each box, if available.
        xywh (torch.Tensor | numpy.ndarray): Boxes in [x, y, width, height] format, calculated on demand.
        xyxyn (torch.Tensor | numpy.ndarray): Normalized [x1, y1, x2, y2] boxes, relative to `orig_shape`.
        xywhn (torch.Tensor | numpy.ndarray): Normalized [x, y, width, height] boxes, relative to `orig_shape`.

    Methods:
        cpu(): Moves the boxes to CPU memory.
        numpy(): Converts the boxes to a numpy array format.
        cuda(): Moves the boxes to CUDA (GPU) memory.
        to(device, dtype=None): Moves the boxes to the specified device.
    """

    def __init__(self, boxes, orig_shape, nc) -> None:
        """
        Initialize the Boxes class.

        Args:
            boxes (torch.Tensor | numpy.ndarray): A tensor or numpy array containing the detection boxes, with
                shape (num_boxes, 6) or (num_boxes, 7). The last two columns contain confidence and class values.
                If present, the third last column contains track IDs.
            orig_shape (tuple): Original image size, in the format (height, width).
        """
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        #assert n in {6, 7}, f"expected 6 or 7 values but got {n}"  # xyxy, track_id, conf, cls
        super().__init__(boxes, orig_shape)
        self.is_track = n == nc
        self.orig_shape = orig_shape
        self.nc = nc

    @property
    def xyxy(self):
        """Return the boxes in xyxy format."""
        return self.data[:, :4]

    @property
    def conf(self):
        """Return the confidence values of the boxes."""
        return self.data[:, -2]

    @property
    def cls(self):
        """Return the class values of the boxes."""
        return self.data[:, -1]

    @property
    def full_conf(self):
        """Return the class values of the boxes."""
        return self.data[:, 4:4+self.nc]
    
    @property
    def id(self):
        """Return the track IDs of the boxes (if available)."""
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)  # maxsize 1 should suffice
    def xywh(self):
        """Return the boxes in xywh format."""
        return ops.xyxy2xywh(self.xyxy)

    @property
    @lru_cache(maxsize=2)
    def xyxyn(self):
        """Return the boxes in xyxy format normalized by original image size."""
        xyxy = self.xyxy.clone() if isinstance(self.xyxy, torch.Tensor) else np.copy(self.xyxy)
        xyxy[..., [0, 2]] /= self.orig_shape[1]
        xyxy[..., [1, 3]] /= self.orig_shape[0]
        return xyxy

    @property
    @lru_cache(maxsize=2)
    def xywhn(self):
        """Return the boxes in xywh format normalized by original image size."""
        xywh = ops.xyxy2xywh(self.xyxy)
        xywh[..., [0, 2]] /= self.orig_shape[1]
        xywh[..., [1, 3]] /= self.orig_shape[0]
        return xywh
    


    def cpu(self):
        """Return a copy of the tensor on CPU memory."""
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.cpu(), self.orig_shape, self.nc)

    def numpy(self):
        """Return a copy of the tensor as a numpy array."""
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy(), self.orig_shape, self.nc)

    def cuda(self):
        """Return a copy of the tensor on GPU memory."""
        return self.__class__(torch.as_tensor(self.data).cuda(), self.orig_shape)

    def to(self, *args, **kwargs):
        """Return a copy of the tensor with the specified device and dtype."""
        return self.__class__(torch.as_tensor(self.data).to(*args, **kwargs), self.orig_shape, self.nc)

    def __len__(self):  # override len(results)
        """Return the length of the data tensor."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return a BaseTensor with the specified index of the data tensor."""
        return self.__class__(self.data[idx], self.orig_shape, self.nc)