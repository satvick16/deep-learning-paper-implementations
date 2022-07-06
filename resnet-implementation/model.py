import torch.nn
import torch
import torch.nn.functional


class Shortcut(torch.nn.module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.conv = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        self.bn = torch.nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ResBlock(torch.nn.module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.act1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(num_features=out_channels)

        if in_channels != out_channels:
            self.shortcut = Shortcut(
                in_channels=in_channels, out_channels=out_channels, stride=stride)
        else:
            self.shortcut = torch.nn.Identity()

    def forward(self, x):
        # F(x, {Wi}) = W2 sigma (W1 x)
        F = self.bn2(self.conv2(self.act1(self.bn1(self.conv1(x)))))
        # y = F(x, {Wi}) + Ws x
        y = F + self.shortcut(x)
        # out = sigma y
        out = self.act2(y)

        return out


class ResNet34(torch.nn.module):
    def __init__(self):
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64,
                            kernel_size=7, stride=2),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),

            ResBlock(64, 64, 1),
            ResBlock(64, 64, 1),
            ResBlock(64, 64, 1),

            ResBlock(64, 128, 2),
            ResBlock(128, 128, 1),
            ResBlock(128, 128, 1),
            ResBlock(128, 128, 1),

            ResBlock(128, 256, 2),
            ResBlock(256, 256, 1),
            ResBlock(256, 256, 1),
            ResBlock(256, 256, 1),
            ResBlock(256, 256, 1),
            ResBlock(256, 256, 1),

            ResBlock(256, 512, 2),
            ResBlock(512, 512, 1),
            ResBlock(512, 512, 1),

            torch.nn.AdaptiveAvgPool2d(output_size=None),
            torch.nn.Linear(in_features=None, out_features=1000),

            torch.nn.Softmax()
        )

        self.apply(self._init_weights)

    def forward(self, x):
        return x

    def _init_weights(self, module):
        # TODO: https://arxiv.org/pdf/1502.01852.pdf
        if isinstance(module, torch.nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

            if module.in_channels == 3:
                module.bias.data.zeros_()
            else:
                module.bias.data.ones_()
        elif isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

            module.bias.data.ones_()
