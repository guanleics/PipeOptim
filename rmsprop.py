# Copyright (c) Lei Guan
# Licensed under the MIT license.

from torch.optim.optimizer import required
from optimizer import OptimizerWithWeightPrediction

class RMSpropWithPipeOptim(OptimizerWithWeightPrediction):
    """
    Weight prediction with RMSprop optimizer.
    """
    def __init__(self, modules, master_parameters, model_parameters,
                 loss_scale, num_versions, lr=required, momentum=0,
                 alpha=0.99, weight_decay=0, verbose_freq=0,
                 partitions=1):
        super(RMSpropWithPipeOptim, self).__init__(
            optim_name='RMSprop',
            modules=modules, master_parameters=master_parameters,
            model_parameters=model_parameters, loss_scale=loss_scale,
            num_versions=num_versions, lr=lr, momentum=momentum,
            alpha=alpha, weight_decay=weight_decay,
            verbose_freq=verbose_freq,
            partitions=partitions,
        )