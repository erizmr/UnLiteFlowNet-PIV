import torch
import torch.nn.functional as F
from src.model.models import Backward, device


def multiscaleUnsupervisorError(tensorFlowForward,
                                tensorFlowBackward,
                                tensorFirst,
                                tensorSecond,
                                weights=None):

    h, w = tensorFirst.shape[-2], tensorFirst.shape[-1]

    def one_scale(tensorFirst, tensorSecond, tensorFlowForward,
                  tensorFlowBackward):
        lambda_s = 3.0
        lambda_c = 0.2
        return bidirectionalDataLoss(tensorFirst, tensorSecond, tensorFlowForward, tensorFlowBackward) \
               + lambda_c * bidirectionalConsistencyLoss(tensorFlowForward, tensorFlowBackward) \
               + lambda_s * biSecondSmoothnessLoss(tensorFlowForward, tensorFlowBackward)

    if weights is None:
        weights = [12.7, 5.5, 4.35, 3.9, 3.4, 1.1]
    assert (len(weights) == len(tensorFlowForward))
    assert (len(weights) == len(tensorFlowBackward))

    loss = 0
    for i, weight in enumerate(weights):

        scale_factor = 1 / (2**i)
        loss += weight * one_scale(tensorFirst, tensorSecond,
                                   tensorFlowForward[-1 - i] * scale_factor,
                                   tensorFlowBackward[-1 - i] * scale_factor)
        h = h // 2
        w = w // 2
        tensorFirst = F.interpolate(tensorFirst, (h, w),
                                    mode='bilinear',
                                    align_corners=False)
        tensorSecond = F.interpolate(tensorSecond, (h, w),
                                     mode='bilinear',
                                     align_corners=False)

    return loss


def charbonnierLoss(x, alpha=0.45, beta=1.0, epsilon=0.001):
    """Compute the generalized charbonnier loss for x
    Args:
        x(tesnor): [batch, channels, height, width]
    Returns:
        loss
    """
    batch, channels, height, width = x.shape
    normalization = torch.tensor(batch * height * width * channels,
                                 requires_grad=False)

    error = torch.pow(
        (x * torch.tensor(beta)).pow(2) + torch.tensor(epsilon).pow(2), alpha)

    return torch.sum(error) / normalization


# photometric difference
def warpLoss(tensorFirst, tensorSecond, tensorFlow):
    """Differentiable Charbonnier penalty function"""
    tensorDifference = tensorFirst - Backward(tensorInput=tensorSecond,
                                              tensorFlow=tensorFlow)
    return charbonnierLoss(tensorDifference, beta=255.0)


def bidirectionalDataLoss(tensorFirst, tensorSecond, tensorFlowForward,
                          tensorFlowBackward):
    """Compute bidirectional photometric loss"""
    return warpLoss(tensorFirst, tensorSecond, tensorFlowForward) + warpLoss(
        tensorSecond, tensorFirst, tensorFlowBackward)


# Flow consistency loss
def consistencyLoss(tensorFlowForward, tensorFlowBackward):
    """ Differentiable Charbonnier penalty function"""
    tensorFlowDifference = tensorFlowForward + Backward(
        tensorInput=tensorFlowBackward, tensorFlow=tensorFlowForward)
    return charbonnierLoss(tensorFlowDifference)


def bidirectionalConsistencyLoss(tensorFlowForward, tensorFlowBackward):
    return consistencyLoss(tensorFlowForward,
                           tensorFlowBackward) + consistencyLoss(
                               tensorFlowBackward, tensorFlowForward)


def _smoothnessDeltas(tensorFlow):
    """1st order smoothness, compute smoothness loss components"""
    out_channels = 2  # u and v
    in_channels = 1  # u or v
    kh, kw = 3, 3

    filter_x = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]])
    filter_y = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]])

    weight = torch.ones(out_channels, in_channels, kh, kw, requires_grad=False)
    weight[0, 0, :, :] = filter_x
    weight[1, 0, :, :] = filter_y

    uFlow, vFlow = torch.split(tensorFlow, split_size_or_sections=1, dim=1)

    delta_u = F.conv2d(uFlow, weight.to(device))
    delta_v = F.conv2d(vFlow, weight.to(device))
    return delta_u, delta_v


def smoothnessLoss(tensorFlow):
    """Compute 1st order smoothness loss"""
    delta_u, delta_v = _smoothnessDeltas(tensorFlow)
    return charbonnierLoss(delta_u) + charbonnierLoss(delta_v)


def bidirectionalSmoothnessLoss(tensorFlowForward, tensorFlowBackward):
    """Compute bidirectional 1st order smoothness loss"""
    return smoothnessLoss(tensorFlowForward) + smoothnessLoss(
        tensorFlowBackward)


# 2nd order smoothness loss
def _secondOrderDeltas(tensorFlow):
    """2nd order smoothness, compute smoothness loss components"""
    out_channels = 4
    in_channels = 1
    kh, kw = 3, 3

    filter_x = [[0, 0, 0], [1, -2, 1], [0, 0, 0]]
    filter_y = [[0, 1, 0], [0, -2, 0], [0, 1, 0]]
    filter_diag1 = [[1, 0, 0], [0, -2, 0], [0, 0, 1]]
    filter_diag2 = [[0, 0, 1], [0, -2, 0], [1, 0, 0]]
    weight = torch.ones(out_channels, in_channels, kh, kw, requires_grad=False)
    weight[0, 0, :, :] = torch.FloatTensor(filter_x)
    weight[1, 0, :, :] = torch.FloatTensor(filter_y)
    weight[2, 0, :, :] = torch.FloatTensor(filter_diag1)
    weight[3, 0, :, :] = torch.FloatTensor(filter_diag2)

    uFlow, vFlow = torch.split(tensorFlow, split_size_or_sections=1, dim=1)
    delta_u = F.conv2d(uFlow, weight.to(device))
    delta_v = F.conv2d(vFlow, weight.to(device))
    return delta_u, delta_v


def secondSmoothnessLoss(tensorFlow):
    """Compute 2nd order smoothness loss"""
    delta_u, delta_v = _secondOrderDeltas(tensorFlow)
    return charbonnierLoss(delta_u) + charbonnierLoss(delta_v)


def biSecondSmoothnessLoss(tensorFlowForward, tensorFlowBackward):
    """Compute bidirectional 2nd order smoothness loss"""
    return secondSmoothnessLoss(tensorFlowForward) + secondSmoothnessLoss(
        tensorFlowBackward)
