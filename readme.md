# Duet: efficient and scalable hybriD neUral rElation undersTanding
## Prepare the Anaconda Environment
1. We recommend Python 3.10.9 with Win10 or Ubuntu
2. ```pip3 install -r requirements.txt```
## Install the sampling algorithm
```python3 ./MySampler/setup.py```

## Dataset Prepare
1. Download DMV dataset used by Naru:https://github.com/naru-project/naru
2. Download Kddcup98 and Census used by UAE:https://github.com/pagegitss/UAE
3. put `Vehicle__Snowmobile__and_Boat_Registrations.csv, cup98.csv, census.csv` into `./datasets`

## Workload Generation
1. run `python3 generate_all_workload_gpu.py`
2. run `python3 generate_train_workload_gpu_npred.py` for queries used to evaluate scalability

## train Duet
* for DMV, run `python3 train_model.py --num-queries=100000 --dataset=dmv --epochs=50 --warmups=12000 --bs=2048 --expand-factor=4 --layers=0 --direct-io --input-encoding=binary --output-encoding=one_hot --multi_pred_embedding=mlp --use-workloads --tag=dmv_mlp_binary_Workloads --gpu-id=0`
* for Kddcup98, run `python3 train_model.py --num-queries=100000 --dataset=cup98 --epochs=50 --warmups=12000 --bs=100 --expand-factor=4 --layers=2 --fc-hiddens=128 --residual --direct-io --input-encoding=binary --output-encoding=one_hot --multi_pred_embedding=mlp --use-workloads --tag=cup98_mlp_binary_Workloads --gpu-id=0`
* for Census, run `python3  train_model.py --num-queries=100000 --dataset=census --epochs=50 --warmups=12000 --bs=100 --expand-factor=4 --layers=2 --fc-hiddens=128 --residual --direct-io --input-encoding=binary --output-encoding=one_hot --multi_pred_embedding=mlp --use-workloads --tag=census_mlp_binary_Workloads --gpu-id=0`
* for Duet's data-driven version, remove the `--use-workloads` option

## evaluate Duet

### Scalability
* run `python3 run_eval_npred.py`
* run `python3 draw_nfilter_curve.py` to draw the scalability plot with the same format as our paper 
### Accuracy
* We give the code to evaluate the error of all epochs and the result of the epoch when the model achieve minium loss
* take DMV as example, run `python3 eval_model.py --dataset=dmv --load_queries=dmv-2000queries-oracle-cards-seed1234.pkl --glob=dmv-16.3MB-data19.550-made-hidden512_256_512_128_1024-emb32-directIo-binaryInone_hotOut-inputNoEmbIfLeq-mlp-seed0 --layers=0 --direct_io --input_encoding=binary --output_encoding=one_hot --multi_pred_embedding=mlp --tag=dmv_mlp_binary_noWorkloads --gpu_id=0 --end_epoch=50`
* for the option `load_queries`, change the seed from 1234 to 42 to switch workloads from Random Queries to In-Workload Queries of the test workload
* for the option `glob`, use the model's name at the format above
* for the rest options, set them according to the training options above