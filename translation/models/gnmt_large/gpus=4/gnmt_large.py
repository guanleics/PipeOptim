# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2
from .stage3 import Stage3

class GNMTSplit(torch.nn.Module):
    def __init__(self):
        super(GNMTSplit, self).__init__()
        self.stage0 = Stage0()
        self.stage1 = Stage1()
        self.stage2 = Stage2()
        self.stage3 = Stage3()

    def forward(self, input0, input1, input2):
        (out0, out2, out1, out3) = self.stage0(input0, input1, input2)
        (out12, out13, out4, out5, out6) = self.stage1(out0, out2, out1, out3)
        (out14, out15, out16, out17) = self.stage2(out12, out13, out4, out5, out6)
        out18 = self.stage3(out12, out14, out15, out16, out17)
        return out18

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
