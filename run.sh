torchrun --standalone --nproc_per_node=2 train.py \
--outdir=/share/huangrenyuan/logs/edm --data=/share/huangrenyuan/dataset/imagenet_64/imagenet-64x64.zip \
--cond=1 --arch=adm --dropout=0.10 --augment=0 --fp16=1 --ls=100 --tick=200 \
--duration=2500 --batch=128 --batch-gpu=32 --lr=1e-4 --ema=50 \
--sigma-learning --pretrain=/share/huangrenyuan/model_zoo/edm/edm-imagenet-64x64-cond-adm.pkl
# --transfer /share/huangrenyuan/model_zoo/edm/edm-imagenet-64x64-cond-adm.pkl
# --sigma-learning --pretrain=/share/huangrenyuan/model_zoo/edm/edm-imagenet-64x64-cond-adm.pkl