import torch.nn.functional as F
import torch


def EPE(input_flow, target_flow, mean=True):
    # Calculate the EPE along the second dimension
    EPE_map = torch.norm(target_flow - input_flow, 2, 1)
    batch_size = EPE_map.size(0)
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum() / batch_size


def realEPE(output, target):
    b, _, h, w = target.size()
    upsampled_output = F.interpolate(output, (h, w),
                                     mode='bilinear',
                                     align_corners=False)
    return EPE(upsampled_output, target, mean=True)
