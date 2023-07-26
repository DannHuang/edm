## 2nd-SED Solver for Diffusion Model</sub>

Abstract: *Diffusion Probabilistic Model (DPM) has achieved remarkable advancement in the area of Image Synthesis. DPM is defined on a forward process where small amount of noise is progressively added to image until the signal is destroyed. A Neural-Network is trained to denoise a white noise back to a real data sample by minimizing the variational Lower-Bound (VLB) of negative log-likelihood, which is called the reverse process. By extending the diffusion length to infinity, both forward and reverse processes can be generalized to Stochastic Differential Equations (SDEs), and these two processes are then integration of the corresponding SDE along time dimension. Current DPM samplers turn out to be some first-order SDE solvers. In this project, we expand the SDE to include higher order terms using It\^o-Taylor expansion, and examine the performance of a second-order SDE solver implemented using forward-mode auto-differentiation in PyTorch.*

## Environments

* Linux and Windows are supported, but the program was implemented solely on Linux so we recommand running on Linux to avoid any unexpected problems.
* 1 high-end NVIDIA GPU (VRAM>12GB) can reproduce the result. We have done all testing and development on 4070Ti.
* 64-bit Python 3.8 and PyTorch 2.0.1 (or later). See https://pytorch.org for PyTorch install instructions.
* Python libraries: See [environment.yml](./environment.yml) for exact library dependencies. You can use the following commands with Anaconda3/Miniconda3 to create and activate your Python environment:
  - `conda env create -f environment.yml -n 2nd_DMSDE`
  - `conda activate 2nd_DMSDE`

## Getting started

To reproduce the main results from our paper, first download checkpoint. Model url is provided in next section:

```.bash
python recompileNN.py --pkl_dir=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl 
```

After the model is recompiled, simply run:

```.bash
python example.py
```

This is a minimal standalone script that loads the best pre-trained model for each dataset and generates a random 8x8 grid of images using the optimal sampler settings. Expected results:

| Dataset  | Runtime | Reference image
| :------- | :------ | :--------------
| CIFAR-10 | ~6 sec  | [`cifar10-32x32.png`](./docs/cifar10-32x32.png)
| FFHQ     | ~28 sec | [`ffhq-64x64.png`](./docs/ffhq-64x64.png)
| AFHQv2   | ~28 sec | [`afhqv2-64x64.png`](./docs/afhqv2-64x64.png)
| ImageNet | ~5 min  | [`imagenet-64x64.png`](./docs/imagenet-64x64.png)

## Pre-trained models

We develop our 2nd-sampler based on pre-trained models from:

**Elucidating the Design Space of Diffusion-Based Generative Models**<br>
Tero Karras, Miika Aittala, Timo Aila, Samuli Laine
<br>https://arxiv.org/abs/2206.00364<br>

- https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
- https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl
- https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl

Feel free to generate more samples with your customize hyper-parameters. The sampler settings can be controlled through command-line options; see [`python generate.py --help`](./docs/generate-help.txt) for more information. For best results, we recommend using the following settings for each dataset:

```.bash
# For CIFAR-10 at 32x32, use deterministic sampling with 84 steps
python generate.py --outdir=Sampler1.2.0/cifar10_N40_rho3_2nd --network=ckpts/edm-cifar10-32x32-cond-vp.pkl --batch=500 --seeds=0-99 --steps=84 --randn_like=ddb --rho=3 --subdirs

# For FFHQ at 64x64, use deterministic sampling with 84 steps
python generate.py --outdir=Sampler1.2.0/ffhq_N84_rho3_2nd --network=ckpts/edm-ffhq-64x64-uncond-vp.pkl --batch=250 --seeds=0-99 --steps=84 --randn_like=ddb --rho=3 --subdirs

# For ImageNet at 64x64, use stochastic sampling with 84 steps
python generate.py --outdir=imgSamples --network=ckpts/edm-imagenet-64x64-cond-adm.pkl --batch=100 --seeds=0-9 --steps=84 --randn_like=ddb --rho=3 --subdirs
```

## Calculating FID

To compute Fr&eacute;chet inception distance (FID) for a given model and sampler, first generate 50,000 random images and then compare them against the dataset reference statistics using `fid.py`:

```.bash
# Generate 50000 images and save them as fid-tmp/*/*.png
python generate.py --outdir=imagenet --network=ckpts/edm-imagenet-64x64-cond-adm.pkl --batch=100 --seeds=0-49999 --steps=84 --randn_like=ddb --rho=3 --subdirs

# Calculate FID
python fid.py calc --images=imgSamples --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz --num=1000
```

Note that the numerical value of FID varies across different random seeds and is highly sensitive to the number of images. By default, `fid.py` will always use 50,000 generated images; providing fewer images will result in an error, whereas providing more will use a random subset. To reduce the effect of random variation, we recommend repeating the calculation multiple times with different seeds, e.g., `--seeds=0-49999`, `--seeds=50000-99999`, and `--seeds=100000-149999`. In our paper, we calculated each FID three times and reported the minimum.

python train.py --outdir=/root/autodl-tmp/imgnet --data=/root/autodl-tmp/ImageNet --cond=1 --arch=adm --precond=sigma --duration=25 --batch=1 --lr=1e-4 --ema=50 --dropout=0.10 --augment=0 --fp16=1 --ls=1 --tick=200 --dm_length=10 --pretrain=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl

python dataset_tool.py --source=/root/autodl-tmp/ImageNet --dest=/root/autodl-tmp/imagenet-64x64.zip --resolution=64x64 --transform=center-crop