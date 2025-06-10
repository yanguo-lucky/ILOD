import torch
from torch.nn import KLDivLoss

class ConsistencyLosses:
    def __init__(self):
        self.kldivloss = KLDivLoss(reduction="none", log_target=False)

    def losses(self,student_roi,teacher_roi):
        loss = {}
        class_scores_student = []
        class_scores_teacher = []
        for s_roi, t_roi in zip (student_roi, teacher_roi):
            class_scores_student.append(s_roi.full_scores) #[:,:-1])
            class_scores_teacher.append(t_roi.full_scores) #[:,:-1])
        class_scores_student=torch.cat(class_scores_student,axis=0)
        class_scores_teacher=torch.cat(class_scores_teacher,axis=0)

        # Weighted KL Divergence
        weights = class_scores_teacher.max(axis=1).values
        kl_loss = self.kldivloss(torch.log(class_scores_student),class_scores_teacher)
        kl_loss = kl_loss.mean(axis=1)*weights
        kl_loss = torch.mean(kl_loss)

        loss['loss_cls_pseudo'] = kl_loss

        return loss
import torch
import torch.nn as nn


def calculate_iaou_loss_batch(pred_boxes_list, target_boxes_list):
    all_pred_boxes = []
    all_target_boxes = []

    device = pred_boxes_list[0].pred_boxes.tensor.device

    for pred_inst, target_inst in zip(pred_boxes_list, target_boxes_list):
        pred_tensor = pred_inst.pred_boxes.tensor
        if pred_tensor.numel() == 0:
            continue

        pred_size = torch.tensor([pred_inst.image_size[1], pred_inst.image_size[0]], dtype=torch.float32, device=device)
        target_size = torch.tensor([target_inst.image_size[1], target_inst.image_size[0]], dtype=torch.float32, device=device)

        norm_pred = pred_tensor.float()
        norm_pred[:, [0, 2]] /= pred_size[0]
        norm_pred[:, [1, 3]] /= pred_size[1]

        target_tensor = target_inst.pred_boxes.float().to(device)
        target_tensor[:, [0, 2]] /= target_size[0]
        target_tensor[:, [1, 3]] /= target_size[1]

        all_pred_boxes.append(norm_pred)
        all_target_boxes.append(target_tensor)

    if len(all_pred_boxes) == 0:
        return torch.tensor(0.0, device=device)

    pred_boxes = torch.cat(all_pred_boxes, dim=0)
    target_boxes = torch.cat(all_target_boxes, dim=0)

    # Calculate intersection
    x_min_inter = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y_min_inter = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x_max_inter = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y_max_inter = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    intersection = torch.clamp(x_max_inter - x_min_inter, min=0) * torch.clamp(y_max_inter - y_min_inter, min=0)

    # Calculate union
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    union = torch.clamp(pred_area + target_area - intersection, min=1e-6)

    # Enclosing box
    x_min_enc = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    y_min_enc = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    x_max_enc = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    y_max_enc = torch.max(pred_boxes[:, 3], target_boxes[:, 3])

    enclosing_area = (x_max_enc - x_min_enc) * (y_max_enc - y_min_enc)
    concave_area = enclosing_area - union

    iaou = (intersection - concave_area) / union

    # Extra penalty term (d1/d2 distance squared)
    width = torch.clamp(x_max_enc - x_min_enc, min=1e-6)
    height = torch.clamp(y_max_enc - y_min_enc, min=1e-6)

    d1 = (pred_boxes[:, 2] - target_boxes[:, 0]) ** 2 + (pred_boxes[:, 1] - target_boxes[:, 3]) ** 2
    d2 = (pred_boxes[:, 0] - target_boxes[:, 2]) ** 2 + (pred_boxes[:, 3] - target_boxes[:, 1]) ** 2

    penalty = (d1 + d2) / (width ** 2 + height ** 2)

    loss = 1 - iaou + penalty
    return loss.mean()


class ConBoxLosses(nn.Module):
    def __init__(self, ciou_weight=0.1):
        super().__init__()
        self.ciou_weight = ciou_weight

    def losses(self, roi_stu, roi_teach):
        iaou_loss = calculate_iaou_loss_batch(roi_stu, roi_teach)
        total_loss = self.ciou_weight * iaou_loss
        return {"loss_box_pseudo": total_loss}

