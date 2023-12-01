# Copyright (c) Lei Guan
# Licensed under the MIT license.

import time
import torch.distributed as dist

def save_image_results(model_name, runtime_list, val_loss_list, val_top1_list, val_top5_list):
    filename = 'experiment_results/' + model_name + \
               '_result_' + time.strftime("%Y%m%d", time.localtime())
    file = open(filename, 'w')
    file.write(str(runtime_list))
    file.write('\r\n')
    file.write(str(val_loss_list))
    file.write('\r\n')
    file.write(str(val_top1_list))
    file.write('\r\n')
    file.write(str(val_top5_list))
    file.write('\r\n')
    file.close()

def save_translation_results(model_name, runtime_list):
    filename = 'experiment_result/' + model_name + \
               '_result_' + time.strftime("%Y%m%d", time.localtime())
    file = open(filename, 'w')
    file.write(str(runtime_list))
    file.write('\r\n')
    file.close()


def show_running_time(epoch, lr, seconds):
    rank = dist.get_rank()
    num_process = dist.get_world_size()
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if rank == num_process-1:
        print("Epoch: %d; lr:%f; Current time: %d:%02d:%02d" % (epoch, lr, h, m, s))




