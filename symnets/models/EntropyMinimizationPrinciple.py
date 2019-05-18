import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"
        

class _Loss(nn.Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=True):
        super(_WeightedLoss, self).__init__(size_average)
        self.register_buffer('weight', weight)


class EMLossForTarget(_WeightedLoss):

    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True, nClass = 10):
        super(EMLossForTarget, self).__init__(weight, size_average)
        self.nClass = nClass

    def forward(self, input):
        batch_size = input.size(0)
        prob = F.softmax(input, dim=1)
        prob_source = prob[:, :self.nClass]
        prob_target = prob[:, self.nClass:]
        prob_sum = prob_target + prob_source
        if (prob_sum.data.cpu() == 0).sum() != 0:
            weight_sum = torch.FloatTensor(batch_size, self.nClass).fill_(0)
            weight_sum[prob_sum.data.cpu() == 0] = 1e-6
            weight_sum = Variable(weight_sum).cuda()
            loss_sum = -(prob_sum + weight_sum).log().mul(prob_sum).sum(1).mean()
        else:
            loss_sum = -prob_sum.log().mul(prob_sum).sum(1).mean()

        return loss_sum