import torch
import torch.nn as nn
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class FocalLoss(nn.Module):
    def __init__(self, n_class, alpha=1, gamma=0):
        super(FocalLoss, self).__init__()
        self.alpha = torch.Tensor([alpha] * n_class)
        self.gamma = gamma

    def forward(self, y_hat, y):
        self.alpha = self.alpha.to(y_hat.device)
        pred_sm = y_hat.softmax(dim=-1)
        pt = pred_sm.gather(dim=1, index=y.view(-1, 1))
        alpha = self.alpha.gather(dim=-1, index=y)
        loss = -alpha * (1 - pt)**self.gamma * pt.log()
        return loss.mean()


class nllloss(nn.Module):
    def __init__(self):
        super(nllloss, self).__init__()
        self.loss_func = nn.NLLLoss()

    def forward(self, y_hat, y):
        return self.loss_func(y_hat, y)

