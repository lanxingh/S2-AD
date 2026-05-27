import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        if inputs.dim() > 2:
            inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, inputs.size(1))
        
        targets = targets.view(-1)
        targets = targets.long()

        inputs = torch.clamp(inputs, 1e-6, 1 - 1e-6)
        log_pt = torch.log(inputs)
        log_pt = log_pt.gather(1, targets.view(-1, 1)).view(-1)
        pt = log_pt.exp()

        loss = -1 * (1 - pt) ** self.gamma * log_pt
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = torch.tensor(self.alpha).to(inputs.device)
                loss = loss * alpha_t

        if self.reduction == 'mean': return loss.mean()
        elif self.reduction == 'sum': return loss.sum()
        else: return loss

class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        N = targets.size()[0]
        smooth = 1
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)
        intersection = input_flat * targets_flat
        dice = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        loss = 1 - dice.sum() / N
        return loss