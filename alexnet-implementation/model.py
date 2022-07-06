import torch.nn
import torch
import torch.nn.functional

'''
references

https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
https://pytorch.org/docs/stable/generated/torch.nn.LocalResponseNorm.html
https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
'''


class AlexNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model = torch.nn.Sequential(
            #### network ####

            torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0,
                            dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None),
            torch.nn.ReLU(inplace=False),
            torch.nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2.0),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0,
                               dilation=1, return_indices=False, ceil_mode=False),

            torch.nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2,
                            dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None),
            torch.nn.ReLU(inplace=False),
            torch.nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2.0),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0,
                               dilation=1, return_indices=False, ceil_mode=False),

            torch.nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1,
                            dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None),
            torch.nn.ReLU(inplace=False),

            torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1,
                            dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None),
            torch.nn.ReLU(inplace=False),

            torch.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1,
                            dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None),
            torch.nn.ReLU(inplace=False),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0,
                               dilation=1, return_indices=False, ceil_mode=False),

            #### classifier ####

            torch.nn.Dropout(p=0.5, inplace=True),
            torch.nn.Linear(in_features=(256 * 6 * 6), out_features=4096,
                            bias=True, device=None, dtype=None),
            torch.nn.ReLU(inplace=False),

            torch.nn.Dropout(p=0.5, inplace=True),
            torch.nn.Linear(in_features=4096, out_features=4096,
                            bias=True, device=None, dtype=None),
            torch.nn.ReLU(inplace=False),

            torch.nn.Linear(in_features=4096, out_features=1000,
                            bias=True, device=None, dtype=None),
            torch.nn.ReLU(inplace=False),

            #### softmax ####

            torch.nn.Softmax(dim=None)
        )

        self.apply(self._init_weights)

    def forward(self, x):
        return self.model(x)

    def _init_weights(self, module):
        # https://gist.github.com/SauravMaheshkar/5704edf87c33ab09033dc9c0a10adaa1
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
