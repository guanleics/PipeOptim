# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch


class Stage0(torch.nn.Module):
    def __init__(self):
        super(Stage0, self).__init__()
        self.layer6 = torch.nn.Embedding(32320, 1024, padding_idx=0)

    def forward(self, input0):
        out0 = input0.clone()
        out6 = self.layer6(out0)
        return (out6)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)
