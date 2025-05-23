# Ultralytics YOLO 🚀, AGPL-3.0 license
import random 

import torch
import torch.nn as nn

from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from ultralytics.utils.metrics import bbox_iou

import numpy as np

from yolo.tnorm import *

# Initialize values for each t-norm
t_norm_values = {
    "product": 0.0,
    "minimum": 0.0,
    "hamacher_product": 0.0,
    "schweizer_sklar": 0.0,
    "lukasiewicz": 0.0,
    "drastic": 0.0,
    "nilpotent_minimum": 0.0,
    "frank": 0.0,
    "yager": 0.0,
    "sugeno_weber": 0.0,
    "aczel_alsina": 0.0,
    "hamacher": 0.0,
}
t_norm_values = {key: float('inf') for key in t_norm_values.keys()}
previous_losses = {key: float('inf') for key in t_norm_values.keys()}

# Learning rate for updating the rl values

# Exploration probability

# Function to select t-norm based on values with ε-greedy exploration
def select_t_norm(beta):
    if random.random() < beta:
        return random.choice(list(t_norm_values.keys()))
    else:
        return max(t_norm_values, key=t_norm_values.get)
    
class MOD_YOLOLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = MOD_YOLOTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        self.constraints = torch.from_numpy(np.load(h.const_path)).to_sparse()
        self.t_norm_usage = {key: 0 for key in t_norm_values.keys()}
        self.t_norm_values = t_norm_values

        self.beta_rl = h.beta_rl
        self.delta_rl = h.delta_rl
        self.rl_mode = h.rl_mode

        if self.rl_mode == "pgl_tnorm":
            for key in list(t_norm_values.keys()):
                if key not in ["product", "minimum", "lukasiewicz"]:
                    del t_norm_values[key], previous_losses[key]
                    
        elif self.rl_mode == "pglhpnmd_tnorm":
            for key in list(t_norm_values.keys()):
                if key not in ["product", "minimum", "lukasiewicz", "hamacher_product", "nilpotent_minimum", "drastic"]:
                    del t_norm_values[key], previous_losses[key]

    def preprocess(self, targets, batch_size, scale_tensor, num_classes):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, num_classes + 4, device=self.device)
        else:
            i = targets[:, 0]  # image index (batch)
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), num_classes + 4, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., :4] = xywh2xyxy(out[..., :4].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):

        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(8, device=self.device)  # box, cls, dfl, req_loss <- Modification point
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["bboxes"], batch["cls"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]], num_classes=self.nc)
        gt_bboxes, gt_labels = targets.split((4, self.nc), 2)  # cls, xyxy
        #targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        #targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        #gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy


        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4+)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        


        with torch.no_grad():
            pred_const = pred_scores.sigmoid()
            pred_const = (pred_const >= 0.3).float().detach()
            if pred_const.sum().item() > 0:
                # Number of boxes that are above the threshold of 0.3 divided by the total number of boxes
                pred_const_max = pred_const.max(-1)[0]
                loss[6] = pred_const_max.mean()
                # Number of labels that are above the threshold of 0.3 divided by the total number of labels
                loss[7] = pred_const.sum() / (pred_const_max > 0).sum()


                pred_const = pred_const[pred_const.sum(-1) > 0]
                pred_const = torch.cat([pred_const, 1-pred_const], axis=-1) # Invert the values
                loss_const = torch.ones((pred_const.shape[0], self.constraints.shape[0]))
                for req_id in range(self.constraints.shape[0]):
                    req_indices = self.constraints.indices()[1][self.constraints.indices()[0]==req_id]
                    loss_const[:,req_id] = 1 - torch.max(pred_const[:,req_indices], axis=-1)[0] # Violation of constraints
                
                loss[4] = (loss_const).mean()
                loss[5] = (loss_const.sum(-1) > 0).float().mean()
                
            else:
                loss[7] = 1
        



        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain


        if self.hyp.req_loss != 0:
            pred_const = pred_scores.sigmoid()
            max_pred = pred_const.max(-1)[0]
            
            max_pred_sorted = torch.sort(max_pred, descending=True, dim=-1)
            
            if self.hyp.req_num_detect == -1:
                pred_const = torch.gather(pred_const, 1, max_pred_sorted[1].unsqueeze(-1).expand(-1, -1, pred_const.shape[-1]))
                pred_const = pred_const[max_pred_sorted[0] > 0.5]
            else:
                pred_const = torch.gather(pred_const, 1, max_pred_sorted[1][:, :self.hyp.req_num_detect].unsqueeze(-1).expand(-1, -1, pred_const.shape[-1]))
                pred_const = pred_const[max_pred_sorted[0][:, :self.hyp.req_num_detect] > 0.5]

            ############################################
            
            if pred_const.shape[0] > 0:
                pred_const = torch.cat([pred_const, 1-pred_const], axis=-1)

                loss_const = torch.ones((pred_const.shape[0], self.constraints.shape[0]), device=self.device)

                if self.hyp.reinforcement_loss:
                    self.hyp.req_type = select_t_norm(self.beta_rl)
                    self.t_norm_usage[self.hyp.req_type] += 1

                for req_id in range(self.constraints.shape[0]):

                    req_ind = self.constraints.indices()[1][self.constraints.indices()[0]==req_id]
                    fuzzy_values = 1 - pred_const[:,req_ind]
                    if self.hyp.req_type == "lukasiewicz":
                        loss_const[:,req_id] = lukasiewicz_tnorm_tensor(fuzzy_values)
                        # loss_const[:,req_id] = apply_tnorm_iterative(lukasiewicz_tnorm, fuzzy_values)
                    elif self.hyp.req_type == "minimum":
                        loss_const[:,req_id] = min_tnorm_tensor(fuzzy_values)
                        # loss_const[:,req_id] = apply_tnorm_iterative(min_tnorm, fuzzy_values)
                    elif self.hyp.req_type == "product":
                        loss_const[:,req_id] = product_tnorm_tensor(fuzzy_values)
                        # loss_const[:,req_id] = apply_tnorm_iterative(product_tnorm, fuzzy_values)
                    elif self.hyp.req_type == "drastic":
                        loss_const[:,req_id] = apply_tnorm_iterative2(drastic_tnorm_batch, fuzzy_values)
                    elif self.hyp.req_type == "nilpotent_minimum":
                        loss_const[:,req_id] = apply_tnorm_iterative2(nilpotentmin_tnorm_batch, fuzzy_values)
                    elif self.hyp.req_type == "hamacher_product":
                        # loss_const[:,req_id] = hamacherprod_tnorm_tensor(fuzzy_values)
                        loss_const[:,req_id] = apply_tnorm_iterative2(hamacherprod_tnorm_batch, fuzzy_values)
                    elif self.hyp.req_type == "yager":
                        loss_const[:,req_id] = apply_tnorm_iterative2(yager_tnorm_batch, fuzzy_values)
                    elif self.hyp.req_type == "frank":
                        loss_const[:,req_id] = apply_tnorm_iterative2(frank_tnorm, fuzzy_values)
                    elif self.hyp.req_type == "sugeno_weber":
                        loss_const[:,req_id] = apply_tnorm_iterative2(sugeno_weber_tnorm, fuzzy_values)
                    # elif self.hyp.req_type == "dombi": # <- Not working
                    #     loss_const[:,req_id] = apply_tnorm_iterative2(dombi_tnorm_batch, fuzzy_values)
                    elif self.hyp.req_type == "aczel_alsina":
                        loss_const[:,req_id] = apply_tnorm_iterative2(aczel_alsina_tnorm_batch, fuzzy_values)
                    elif self.hyp.req_type == "hamacher":
                        loss_const[:,req_id] = apply_tnorm_iterative2(hamacher_tnorm, fuzzy_values)
                    elif self.hyp.req_type == "schweizer_sklar":
                        loss_const[:,req_id] = apply_tnorm_iterative2(schweizer_sklar_tnorm_batch, fuzzy_values)
                        
                    else:
                        raise ValueError
                
                if torch.isnan(loss_const).any():
                    print("Nan values in loss_const")
                    torch.nan_to_num(loss_const, nan=0.0)
                    exit()
                current_loss = loss_const.sum() / (loss_const.shape[0] * loss_const.shape[1])

                # Check if current_loss is nan
                if torch.isnan(current_loss):
                    current_loss = current_loss * 0
                    import csv
                    with open('losses.csv', 'w') as file:
                        writer = csv.writer(file)
                        writer.writerow([self.hyp.req_type, current_loss.item()])

                loss[3] = current_loss * self.hyp.req_loss
                
                if self.rl_mode == "c_violation":
                    current_loss = loss[4]
                elif self.rl_mode == "all_loss":
                    current_loss = loss[:4].sum()

                if self.hyp.reinforcement_loss:
                    # first_update = True
                    epsilon = 1e-9

                    previous_loss = previous_losses[self.hyp.req_type]
                    
                    if previous_loss == float('inf'):
                        previous_losses[self.hyp.req_type] = current_loss.item()
                    else:
                        if previous_loss < epsilon:
                            previous_loss = epsilon                            
                                
                        normalized_update = (previous_loss - current_loss.item()) / (previous_loss)

                        if t_norm_values[self.hyp.req_type] == float('inf'):
                            t_norm_values[self.hyp.req_type] = self.delta_rl * normalized_update
                        else:
                            # t_norm_values[self.hyp.req_type] += delta_rl * normalized_update
                            t_norm_values[self.hyp.req_type] = self.delta_rl * t_norm_values[self.hyp.req_type] + (1 - self.delta_rl) * normalized_update
                        previous_losses[self.hyp.req_type] = current_loss.item()

        
        # loss[3] *= self.hyp.req_loss # req_loss gain
        # def is_main_process():
        #     import torch.distributed as dist
        #     return not dist.is_initialized() or dist.get_rank() == 0
        
        # if self.hyp.reinforcement_loss and is_main_process:
        #     import json
        #     with open(f"t_norm_usage.txt", 'a+') as t_norm_usage_file:
        #         t_norm_usage_file.write('\n')
        #         t_norm_usage_file.write(json.dumps(self.t_norm_usage))
        
        self.t_norm_values = t_norm_values
        return loss[:4].sum() * batch_size, loss.detach()  # loss(box, cls, dfl, rql)





class MOD_YOLOTaskAlignedAssigner(nn.Module):
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters."""
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment. Reference code is available at
        https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py.

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, num_classes)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)

        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
            target_gt_idx (Tensor): shape(bs, num_total_anchors)
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx).to(device),
                torch.zeros_like(pd_bboxes).to(device),
                torch.zeros_like(pd_scores).to(device),
                torch.zeros_like(pd_scores[..., 0]).to(device),
                torch.zeros_like(pd_scores[..., 0]).to(device),
            )

        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """Get in_gts mask, (b, max_num_obj, h*w)."""
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment metric given predicted and ground truth bounding boxes."""
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        #ind[1] = gt_labels.squeeze(-1)[:, :, 0]  # b, max_num_obj (class per obj)
        # Get the scores of each grid for each gt cls
        # pd_scores[ind[0]]: b, h*w, num_class (b, max_num_obj) -> b, max_num_obj, h*w, num_class
        # pd_scores[ind[0]]: b, h*w, num_class (b, max_num_obj), (b, max_num_obj) -> b, max_num_obj, h*w, num_class

        #bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w
        bbox_scores[mask_gt] = (torch.mean(pd_scores[ind[0]] * gt_labels.unsqueeze(-2).repeat(1,1,na,1), dim=-1).to(dtype=pd_scores.dtype))[mask_gt]

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """IoU calculation for horizontal bounding boxes."""
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Select the top-k candidates based on the given metrics.

        Args:
            metrics (Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size,
                              max_num_obj is the maximum number of objects, and h*w represents the
                              total number of anchor points.
            largest (bool): If True, select the largest values; otherwise, select the smallest values.
            topk_mask (Tensor): An optional boolean tensor of shape (b, max_num_obj, topk), where
                                topk is the number of top candidates to consider. If not provided,
                                the top-k values are automatically computed based on the given metrics.

        Returns:
            (Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """

        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        # count_tensor.scatter_add_(-1, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int8, device=topk_idxs.device))
        # Filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the
                                batch size and max_num_obj is the maximum number of objects.
            gt_bboxes (Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (Tensor): Indices of the assigned ground truth objects for positive
                                    anchor points, with shape (b, h*w), where h*w is the total
                                    number of anchor points.
            fg_mask (Tensor): A boolean tensor of shape (b, h*w) indicating the positive
                              (foreground) anchor points.

        Returns:
            (Tuple[Tensor, Tensor, Tensor]): A tuple containing the following tensors:
                - target_labels (Tensor): Shape (b, h*w), containing the target labels for
                                          positive anchor points.
                - target_bboxes (Tensor): Shape (b, h*w, 4), containing the target bounding boxes
                                          for positive anchor points.
                - target_scores (Tensor): Shape (b, h*w, num_classes), containing the target scores
                                          for positive anchor points, where num_classes is the number
                                          of object classes.
        """

        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        #target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)
        target_labels = gt_labels.long()[:, :, 0].flatten()[target_gt_idx]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()
        #target_scores = torch.zeros(
        #    (target_labels.shape[0], target_labels.shape[1], self.num_classes),
        #    dtype=torch.int64,
        #    device=target_labels.device,
        #)  # (b, h*w, 80)
        #target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        target_scores = gt_labels.view(-1, self.num_classes)[target_gt_idx]

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
        """
        Select the positive anchor center in gt.

        Args:
            xy_centers (Tensor): shape(h*w, 2)
            gt_bboxes (Tensor): shape(b, n_boxes, 4)

        Returns:
            (Tensor): shape(b, n_boxes, h*w)
        """
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        # return (bbox_deltas.min(3)[0] > eps).to(gt_bboxes.dtype)
        return bbox_deltas.amin(3).gt_(eps)

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """
        If an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.

        Args:
            mask_pos (Tensor): shape(b, n_max_boxes, h*w)
            overlaps (Tensor): shape(b, n_max_boxes, h*w)

        Returns:
            target_gt_idx (Tensor): shape(b, h*w)
            fg_mask (Tensor): shape(b, h*w)
            mask_pos (Tensor): shape(b, n_max_boxes, h*w)
        """
        # (b, n_max_boxes, h*w) -> (b, h*w)
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)
            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)

            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)
            fg_mask = mask_pos.sum(-2)
        # Find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos



def int2binary(x, bits):
    mask = 2**torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()



from ultralytics.utils.tal import bbox2dist
import torch.nn.functional as F


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)