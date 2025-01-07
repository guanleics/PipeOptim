NUM_GPU=4
MASTER_ADDR=localhost
DECAY=0.0005
EPOCH=3
LR=0.001
dataset=/data/home/djy24/dataset

BS=295

python main_with_runtime_cifar_throughput.py --optim adamw --module models.googlenet.gpus=4_straight -b $BS --data_dir $dataset --config_path models/googlenet/gpus=4_straight/mp_conf_${NUM_GPU}gpu.json --distributed_backend gloo --master_addr $MASTER_ADDR --epochs ${EPOCH} --lr $LR --weight-decay $DECAY --print-freq 200 --log_dir logs/xpipe_${NUM_GPU}_lr${LR} --lr_policy step  --rank 0 --local_rank 0 --world_size 4  &
python main_with_runtime_cifar_throughput.py --optim adamw --module models.googlenet.gpus=4_straight -b $BS --data_dir $dataset --config_path models/googlenet/gpus=4_straight/mp_conf_${NUM_GPU}gpu.json --distributed_backend gloo --master_addr $MASTER_ADDR --epochs ${EPOCH} --lr $LR --weight-decay $DECAY --print-freq 200 --log_dir logs/xpipe_${NUM_GPU}_lr${LR} --lr_policy step  --rank 1 --local_rank 1 --world_size 4  &
python main_with_runtime_cifar_throughput.py --optim adamw --module models.googlenet.gpus=4_straight -b $BS --data_dir $dataset --config_path models/googlenet/gpus=4_straight/mp_conf_${NUM_GPU}gpu.json --distributed_backend gloo --master_addr $MASTER_ADDR --epochs ${EPOCH} --lr $LR --weight-decay $DECAY --print-freq 200 --log_dir logs/xpipe_${NUM_GPU}_lr${LR} --lr_policy step  --rank 2 --local_rank 2 --world_size 4  &
python main_with_runtime_cifar_throughput.py --optim adamw --module models.googlenet.gpus=4_straight -b $BS --data_dir $dataset --config_path models/googlenet/gpus=4_straight/mp_conf_${NUM_GPU}gpu.json --distributed_backend gloo --master_addr $MASTER_ADDR --epochs ${EPOCH} --lr $LR --weight-decay $DECAY --print-freq 200 --log_dir logs/xpipe_${NUM_GPU}_lr${LR} --lr_policy step  --rank 3 --local_rank 3 --world_size 4 \
 >> ./experiment_results/pipeoptim-p1-cifar10-googlenet_adamw_throughput_bs295.txt