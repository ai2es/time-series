import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


tol = torch.finfo(torch.float32).eps


class LinearNormalGamma(nn.Module):
    def __init__(self, in_chanels, out_channels):
        super().__init__()
        self.linear = nn.utils.spectral_norm(nn.Linear(in_chanels, out_channels*4))

    def evidence(self, x):
        return  torch.log(torch.clamp(torch.exp(x), min=0) + 1)

    def forward(self, x):
        pred = self.linear(x).view(x.shape[0], -1, 4)
        mu, logv, logalpha, logbeta = [w.squeeze(-1) for w in torch.split(pred, 1, dim=-1)]
        return mu, self.evidence(logv), self.evidence(logalpha) + 1, self.evidence(logbeta)


def nig_nll(y, gamma, v, alpha, beta):
    two_blambda = 2 * beta * (1 + v) + tol
    nll = 0.5 * torch.log(np.pi / (v + tol)) \
            - alpha * torch.log(two_blambda + tol) \
            + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + two_blambda + tol) \
            + torch.lgamma(alpha) \
            - torch.lgamma(alpha + 0.5)

    return nll


def nig_reg(y, gamma, v, alpha, beta):
    error = F.l1_loss(y, gamma, reduction="none")
    evi = 2 * v + alpha
    return error * evi


def evidential_regresssion_loss(y, pred, coeff=1.0):
    gamma, v, alpha, beta = pred
    loss_nll = nig_nll(y, gamma, v, alpha, beta)
    loss_reg = nig_reg(y, gamma, v, alpha, beta)
    return loss_nll.mean() + coeff * loss_reg.mean()