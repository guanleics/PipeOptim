# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch


class Stage8(torch.nn.Module):
    def __init__(self):
        super(Stage8, self).__init__()
        self.layer3 = torch.nn.Dropout(p=0.2)
        self.layer5 = torch.nn.LSTM(2048, 1024)

    def forward(self, input1, input0):
        out0 = input0.clone()
        out1 = input1.clone()
        out2 = None
        out3 = self.layer3(out0)
        out4 = torch.cat([out3, out1], 2)
        out5 = self.layer5(out4, out2)
        return (out5[0], out0)

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
