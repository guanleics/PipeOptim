NUM_GPU=4
MASTER_ADDR=localhost
DECAY=0.0005
EPOCH=1
BS=130  #120
dataset=/data/home/djy24/dataset/wmt16_de_en

python main_with_runtime_throughput.py --data_dir $dataset --module models.gnmt_large.gpus=4 --config_path models/gnmt_large/gpus=4/mp_conf.json  --master_addr localhost  --distributed_backend gloo --lr 0.0003 -b $BS  --epochs ${EPOCH} --print-freq 200 --checkpoint_dir ./checkpoint_dir --recompute --partitions 1 --rank 0 --local_rank 0 &
python main_with_runtime_throughput.py --data_dir $dataset --module models.gnmt_large.gpus=4 --config_path models/gnmt_large/gpus=4/mp_conf.json  --master_addr localhost  --distributed_backend gloo --lr 0.0003 -b $BS  --epochs ${EPOCH} --print-freq 200 --checkpoint_dir ./checkpoint_dir --recompute --partitions 1 --rank 1 --local_rank 1 &
python main_with_runtime_throughput.py --data_dir $dataset --module models.gnmt_large.gpus=4 --config_path models/gnmt_large/gpus=4/mp_conf.json  --master_addr localhost  --distributed_backend gloo --lr 0.0003 -b $BS  --epochs ${EPOCH} --print-freq 200 --checkpoint_dir ./checkpoint_dir --recompute --partitions 1 --rank 2 --local_rank 2 &
python main_with_runtime_throughput.py --data_dir $dataset --module models.gnmt_large.gpus=4 --config_path models/gnmt_large/gpus=4/mp_conf.json  --master_addr localhost  --distributed_backend gloo --lr 0.0003 -b $BS  --epochs ${EPOCH} --print-freq 200 --checkpoint_dir ./checkpoint_dir --recompute --partitions 1 --rank 3 --local_rank 3 \
 >> ./experiment_results/xpipe_gnmt16_throughput_bs130.txt
