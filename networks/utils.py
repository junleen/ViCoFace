from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
import os
import collections

class Logger:
    def __init__(self, ):

        self.loss_list = {}
        self.names = None
        self.iter = 0

    def reset(self):
        self.loss_list = {}

    def get_losses(self):
        losses = {}
        for k, v in self.loss_list.items():
            losses[k] = np.array(v).mean()
        return losses
    
    def get_loss_str(self):
        losses = self.get_losses()
        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in losses.items()])
        return loss_string
    
    def log_iter(self, losses):
        self.iter += 1
        losses = collections.OrderedDict(losses.items())
        for k, v in losses.items():
            if not k.endswith('loss'):
                continue
            if k in self.loss_list:
                self.loss_list[k].append(v.item())
            else:
                self.loss_list[k] = [v.item()]
        if self.names is None:
            self.names = list(losses.keys())

class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """

    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out
