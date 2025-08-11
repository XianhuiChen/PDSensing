import torch
from torch import nn
import torch.nn.functional as F


class TAMELoss(nn.Module):
    def __init__(self):
        super(TAMELoss, self).__init__()
        self.classify_loss = nn.BCELoss()

    def forward(self, prob, labels, train=True):
        assert len(prob) > 0
        pos_ind = labels > 0.5
        neg_ind = labels < 0.5
        pos_label = labels[pos_ind]
        neg_label = labels[neg_ind]
        pos_prob = prob[pos_ind]
        neg_prob = prob[neg_ind]
        pos_loss, neg_loss = 0, 0
        if len(pos_prob):
            pos_loss = 0.5 * self.classify_loss(pos_prob, pos_label)
        if len(neg_prob):
            neg_loss = 0.5 * self.classify_loss(neg_prob, neg_label)
        classify_loss = pos_loss + neg_loss
        return classify_loss


def cal_predictor_loss(pred, label, mask, mask_next, device, phase):
    """
    Calculates the loss and the number of correct predictions for a batch.

    Args:
        pred (torch.Tensor): Predicted probabilities (shape: [batch_size, 1]).
        label (torch.Tensor): Ground truth labels (shape: [batch_size]).
        device (torch.device): Device to perform computations (CPU or GPU).

    Returns:
        tuple:
            - loss (torch.Tensor): The calculated loss for the batch.
            - n_correct (int): Number of correct predictions in the batch.
    """
    label = label.float()
    if len(pred.size()) > 3:
        assert len(label.size()) == 1
        label_shape = [len(label)] + [1 for _ in pred.size()][1:]
        label = label.view(label_shape).expand_as(pred)
    prediction_loss = TAMELoss()(pred, label)
    return prediction_loss


def cal_actor_loss(action_pred, action_gt_index, mask):
    """
    action_pred: [B, W, T, H, 4] - logits
    action_gt: [B, W, T, H, 4] - one-hot or label index in one-hot format
    mask: [B, W, T, H, 4] - binary mask per action
    """

    B, W, T, H, A = action_pred.shape
    action_pred_flat = action_pred.view(-1, A)
    action_gt_index_flat = action_gt_index.view(-1)
    mask_flat = mask.view(-1, A)
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    pred_loss = loss_fn(action_pred_flat, action_gt_index_flat)
    loss_all = pred_loss
    loss_per_action = torch.zeros_like(mask_flat, dtype=loss_all.dtype)
    loss_per_action[torch.arange(loss_all.size(0)), action_gt_index_flat] = loss_all
    valid_loss = loss_per_action * mask_flat
    valid_counts = torch.sum(mask_flat, dtype=torch.float32)
    final_loss = torch.sum(valid_loss) / (valid_counts + 1e-6)
    return final_loss