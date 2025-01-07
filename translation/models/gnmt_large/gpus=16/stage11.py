# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch


class Stage11(torch.nn.Module):
    def __init__(self):
        super(Stage11, self).__init__()
        self.layer5 = torch.nn.LSTM(2048, 1024)
        self.layer8 = torch.nn.Dropout(p=0.2)

    def forward(self, input4, input3, input0):
        out0 = input0.clone()
        out1 = input4.clone()
        out2 = None
        out3 = input3.clone()
        out4 = torch.cat([out0, out1], 2)
        out5 = self.layer5(out4, out2)
        out6 = out5[0]
        out6 = out6 + out3
        out8 = self.layer8(out6)
        out9 = torch.cat([out8, out1], 2)
        return (out6, out9)

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
