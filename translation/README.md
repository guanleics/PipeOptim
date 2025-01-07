### PipeDream Command Lines (for Translation Tasks)
1) create docker container
```bash
cd /home/lei/pipedream-pipedream
docker build --tag lei:lei .
```
2) run docker
```bash
nvidia-docker run -it -v /home/:/home/ --ipc=host --net=host lei:lei /bin/bash
```
3) Train GNMT-8
```buildoutcfg
cd /home/lei/pipedream-pipedream/runtime/translation
sudo /opt/conda/bin/python setup.py install
/opt/conda/bin/python main_with_runtime.py --data_dir /home/data/wmt16_de_en --module models.gnmt.gpus=4 \
 --config_path models/gnmt/gpus=4/mp_conf.json  --master_addr localhost  --distributed_backend gloo \
  --lr 0.0003 -b 64  --epochs 15 --print-freq 200 --checkpoint_dir ./checkpoint_dir --recompute \
   --rank 0 --local_rank 0
```
4) Train GNMT-16
```buildoutcfg
cd /home/lei/pipedream-pipedream/runtime/translation
sudo /opt/conda/bin/python setup.py install
/opt/conda/bin/python main_with_runtime.py --data_dir /home/data/wmt16_de_en --module models.gnmt_large.gpus=4 --config_path models/gnmt_large/gpus=4/mp_conf.json  --master_addr localhost  --distributed_backend gloo --lr 0.0003 -b 64 --epochs 15 --print-freq 2000 --checkpoint_dir ./checkpoint_dir --recompute --rank 0 --local_rank 0 >> pipedream_gnmt_large0
```
5) Compute BLEU GNMT-8 & GNMT-16
```bash
/opt/conda/bin/python compute_bleu_scores.py \
--dataset_dir /home/data/wmt16_de_en --num_stages 4 --input /home/data/wmt16_de_en/newstest2014.tok.bpe.32000.en --reference /home/data/wmt16_de_en/newstest2014.de --output ./output_dir/output.txt --checkpoint_path /home/data/checkpoint_ori --module models.gnmt.gpus=4 --num_layers 4 --math fp32
```

```bash
/opt/conda/bin/python compute_bleu_scores.py --dataset_dir /home/data/wmt16_de_en --num_stages 4 --input /home/data/wmt16_de_en/newstest2014.tok.bpe.32000.en --reference /home/data/wmt16_de_en/newstest2014.de --output ./output_dir/output.txt --checkpoint_path ./checkpoint_dir --module models.gnmt_large.gpus=4 --num_layers 8 --math fp32
```

/opt/conda/bin/python compute_bleu_scores.py --dataset_dir /home/data/wmt16_de_en --num_stages 4 --input /home/data/wmt16_de_en/newstest2014.tok.bpe.32000.en --reference /home/data/wmt16_de_en/newstest2014.de --output ./output_dir/output.txt --checkpoint_path /home/data/checkpoint_ori --module models.gnmt.gpus=4 --num_layers 4 --math fp32

/opt/conda/bin/python main_with_runtime_throughput.py --data_dir /home/data/wmt16_de_en --module models.gnmt.gpus=4 --config_path models/gnmt/gpus=4/mp_conf.json  --master_addr localhost  --distributed_backend gloo --lr 0.0003 -b 100  --epochs 1 --print-freq 200 --checkpoint_dir ./checkpoint_dir --recompute --rank 0 --local_rank 0

