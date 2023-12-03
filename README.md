# PipeOptim
A PyTorch implementation of PipeOptim


## Introduction
This is the official implementation of [PipeOptim: Ensuring Effective 1F1B Schedule with Optimizer-Dependent Weight Prediction] (PipeOptim).

This is a PyTroch implementation (based on the source code of [PipeDream](https://github.com/msr-fiddle/pipedream) and [SpecTrain](https://github.com/ntueclab/SpecTrain-PyTorch)).

The following figure describes the main idea of PipeOptim.

<div align="center">
<img src="fig/pipeoptim.jpeg" alt="drawing" width="600" />
</div>

PipeOptim uses the predicted weights to perform forward. The prediction formula for the forward pass is: 
```math
\hat{\mathbf W}_{t+s} = \mathbf W_t - lr \cdot s \cdot\Delta \mathbf{W}_{t},
```
where $lr$ is the learning rate, $s$ denotes the weight version difference, and $\Delta \mathbf{W}_{t}$ are computed based on the update rule of the used optimizer.

PipeOptim lets each GPU compute $s$ via
```math
	s= D -rank -1, 
```
where $D$ refers to the pipeline depth and $rank$ is the index of a stage with $rank \in \{0, 1, \dots, size-1\}$.

PipeOptim computes $\Delta \mathbf{W}_{t}$ according to the update rule of the used optimizer.


## Environmental Setup
The experiment settings are the same as [PipeDream](https://github.com/msr-fiddle/pipedream).


## Quick Start
```bash
cd PipeOptim/image_classification
bash scripts/resnet/pipeoptim.sh


## License
Copyright (c) Lei Guan, Dongsheng Li, Jiye Liang, Wenjian Wang, Xicheng Lu. All rights reserved.
Licensed under the [MIT](LICENSE.txt) license.
