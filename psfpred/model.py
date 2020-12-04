import torch
import torch.nn.functional as F


class ResBlock1D(torch.nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        stride=1,
        dilation=1,
    ):
        super(ResBlock1D, self).__init__()

        padding = kernel_size // 2

        self.conv1 = torch.nn.Conv1d(channels,
                                     channels,
                                     kernel_size=(kernel_size),
                                     dilation=(dilation),
                                     stride=(stride),
                                     padding=(padding),
                                     bias=False)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.activation = F.relu
        self.conv2 = torch.nn.Conv1d(channels,
                                     channels,
                                     kernel_size=(kernel_size),
                                     dilation=(dilation),
                                     stride=(stride),
                                     padding=(padding),
                                     bias=False)
        self.bn2 = torch.nn.BatchNorm1d(channels)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += x
        out = self.activation(out)
        return out


class ResBlock2D(torch.nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        stride=1,
        dilation=1,
    ):
        super(ResBlock2D, self).__init__()

        padding = kernel_size // 2

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


class Model(torch.nn.Module):
    def __init__(self, max_seq_len=200, num_classes=(50 + 1)):
        super(Model, self).__init__()

        num_aa_types = 21
        kernel_size_1d = 11
        kernel_size_2d = 5

        num_1d_blocks = 3
        self.seq_branch = torch.nn.Sequential(*[
            ResBlock1D(num_aa_types, kernel_size_1d)
            for _ in range(num_1d_blocks)
        ])

        dist_branch_channels = 22
        self.dist_branch = torch.nn.Conv2d(1,
                                           dist_branch_channels,
                                           kernel_size=(kernel_size_2d,
                                                        kernel_size_2d),
                                           padding=(kernel_size_2d // 2,
                                                    kernel_size_2d // 2),
                                           dilation=(1, 1),
                                           stride=1)

        num_2d_blocks = 10
        main_branch_channels = 2 * num_aa_types + dist_branch_channels
        self.main_branch = torch.nn.Sequential(
            *[
                ResBlock2D(main_branch_channels, kernel_size_2d)
                for _ in range(num_2d_blocks)
            ],
            torch.nn.Conv2d(main_branch_channels,
                            num_classes,
                            kernel_size=(kernel_size_2d, kernel_size_2d),
                            padding=(kernel_size_2d // 2, kernel_size_2d // 2),
                            dilation=(1, 1),
                            stride=1))

        maxpool_kernel_size = 5
        self.maxpool_net = torch.nn.Sequential(*[
            torch.nn.MaxPool2d(maxpool_kernel_size),
            torch.nn.Conv2d(num_classes,
                            num_classes,
                            kernel_size=(kernel_size_2d, kernel_size_2d),
                            padding=(kernel_size_2d // 2, kernel_size_2d // 2),
                            dilation=(1, 1),
                            stride=1),
            torch.nn.MaxPool2d(maxpool_kernel_size),
            torch.nn.Conv2d(num_classes,
                            num_classes,
                            kernel_size=(kernel_size_2d, kernel_size_2d),
                            padding=(kernel_size_2d // 2, kernel_size_2d // 2),
                            dilation=(1, 1),
                            stride=1),
            torch.nn.MaxPool2d(max_seq_len // (maxpool_kernel_size**2))
        ])

    def forward(self, seq_input, dist_input):
        out_1d = self.seq_branch(seq_input)
        expand_out_1d = out_1d.unsqueeze(-1).expand(
            (*out_1d.shape, out_1d.shape[-1]))
        seq_out_2d = torch.cat(
            [expand_out_1d, expand_out_1d.transpose(2, 3)], dim=1)

        dist_out_2d = self.dist_branch(dist_input)

        main_input = torch.cat([seq_out_2d, dist_out_2d], dim=1)
        main_out = self.main_branch(main_input)

        out = self.maxpool_net(main_out)
        out = out.squeeze(-1).squeeze(-1)

        return out
