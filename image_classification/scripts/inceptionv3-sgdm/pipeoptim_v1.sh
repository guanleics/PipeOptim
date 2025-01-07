NUM_GPU=4
MASTER_ADDR=localhost
DECAY=0.0005
EPOCH=70
LR=0.01
BS=64 #78 #
#cdataset=/gf3/home/lei/dataset/tiny-imagenet-200
dataset=/data/home/djy24/dataset/tiny-imagenet-200


python main_with_runtime.py --module models.inceptionv3.gpus=4 -b $BS --data_dir $dataset --config_path models/inceptionv3/gpus=4/mp_conf.json --distributed_backend gloo --master_addr $MASTER_ADDR --epochs ${EPOCH} --lr $LR --weight-decay $DECAY --print-freq 200 --log_dir logs/xpipe_${NUM_GPU}_lr${LR} --lr_policy step --rank 0 --local_rank 0 --world_size 4  &
python main_with_runtime.py --module models.inceptionv3.gpus=4 -b $BS --data_dir $dataset --config_path models/inceptionv3/gpus=4/mp_conf.json --distributed_backend gloo --master_addr $MASTER_ADDR --epochs ${EPOCH} --lr $LR --weight-decay $DECAY --print-freq 200 --log_dir logs/xpipe_${NUM_GPU}_lr${LR} --lr_policy step --rank 1 --local_rank 1 --world_size 4  &
python main_with_runtime.py --module models.inceptionv3.gpus=4 -b $BS --data_dir $dataset --config_path models/inceptionv3/gpus=4/mp_conf.json --distributed_backend gloo --master_addr $MASTER_ADDR --epochs ${EPOCH} --lr $LR --weight-decay $DECAY --print-freq 200 --log_dir logs/xpipe_${NUM_GPU}_lr${LR} --lr_policy step --rank 2 --local_rank 2 --world_size 4  &
python main_with_runtime.py --module models.inceptionv3.gpus=4 -b $BS --data_dir $dataset --config_path models/inceptionv3/gpus=4/mp_conf.json --distributed_backend gloo --master_addr $MASTER_ADDR --epochs ${EPOCH} --lr $LR --weight-decay $DECAY --print-freq 200 --log_dir logs/xpipe_${NUM_GPU}_lr${LR} --lr_policy step --rank 3 --local_rank 3 --world_size 4 \
 >> ./experiment_results/xpipe-p1-inceptionv3_tinyimagenet.txt
