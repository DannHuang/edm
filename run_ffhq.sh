torchrun --standalone --nproc_per_node=2 train.py \
--outdir /share/huangrenyuan/logs/ffhq \
--data /share/huangrenyuan/dataset/ffhq/imagenet/ffhq-64x64.zip \
--cond=0 --arch=ncsnpp --cres=1,2,2,2 --lr=2e-4 --dropout=0.05 --augment=0.15 \
--batch=256 --batch-gpu=64 --tick=12 \
--sigma-learning --pretrain=/share/huangrenyuan/model_zoo/edm/edm-ffhq-64x64-uncond-ve.pkl --sigma-lr=1e-2 \