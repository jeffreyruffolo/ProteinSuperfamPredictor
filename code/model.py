import torch


class ResBlock2D(torch.nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        padding,
        stride=1,
        dilation=1,
    ):
        super(ResBlock2D, self).__init__()

        self.conv1 = torch.nn.Conv2d(channels,
                                     channels,
                                     kernel_size=(kernel_size, kernel_size),
                                     dilation=(dilation, dilation),
                                     stride=(stride, stride),
                                     padding=(padding, padding),
                                     bias=False)
        self.bn1 = torch.nn.BatchNorm2d(channels)
        self.activation = F.relu
        self.conv2 = torch.nn.Conv2d(channels,
                                     channels,
                                     kernel_size=(kernel_size, kernel_size),
                                     dilation=(dilation, dilation),
                                     stride=(stride, stride),
                                     padding=(padding, padding),
                                     bias=False)
        self.bn2 = torch.nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += x
        out = self.activation(out)
        return out