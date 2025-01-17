# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc

#----------------------------------------------------------------------------
class sigmoid_model(nn.Module):
    def __init__(self, dm_length=10, sigma_max=80.0, sigma_min=0.002, rho=3.0):
        super().__init__()
        self.length=dm_length
        self.sigma_max=sigma_max
        self.sigma_min=sigma_min
        self.rho=rho
        A = torch.tensor(self.sigma_max**(1/self.rho), dtype=torch.float64)
        B = torch.tensor(self.sigma_min**(1/self.rho) - self.sigma_max**(1/self.rho), dtype=torch.float64)
        # # Time step discretization, turn time-steps into sigma-schedule.
        step_indices = torch.arange(self.length, dtype=torch.float64)  # [0,1,...,num_steps-1]
        sigmas = (A + step_indices/(self.length-1)*B).pow(self.rho)
        ratio = sigmas[1:]/sigmas[:-1]
        logits=-(1/ratio-1).log()
        # # EDM init
        self.init_vec=logits
        # # Random init
        # self.init_vec=torch.randn(self.length, dtype=torch.float64)
        # # Resume previous
        # self.init_vec=torch.tensor([9.8277, 9.5160, 9.7185, 6.8753, 7.8725, 5.9169, 5.2962, 5.0081, 4.1049, 2.2808], dtype=torch.float32)
        self.vec=torch.nn.Parameter(self.init_vec)

    def forward(self):
        # sigmoid model
        return F.sigmoid(self.vec)
    
    def sigmas(self):
        lambdas=self.forward()
        ratio=torch.cat([torch.ones_like(lambdas[:1])*self.sigma_max, lambdas])
        sigmas=torch.cumprod(ratio, dim=0)
        # sigmas=torch.cat([sigmas, torch.ones_like(lambdas[:1])*sigma_min])
        return sigmas

class softmax_model(nn.Module):
    def __init__(self, dm_length=10, sigma_max=80.0, sigma_min=0.002, rho=3.0):
        super().__init__()
        self.length=dm_length-2
        self.sigma_max=sigma_max
        self.sigma_min=sigma_min
        self.rho=rho
        A = torch.tensor(self.sigma_max**(1/self.rho), dtype=torch.float64)
        B = torch.tensor(self.sigma_min**(1/self.rho) - self.sigma_max**(1/self.rho), dtype=torch.float64)
        # # Time step discretization, turn time-steps into sigma-schedule.
        step_indices = torch.arange(dm_length, dtype=torch.float64)  # [0,1,...,num_steps-1]
        sigmas = (A + step_indices/(dm_length-1)*B).pow(self.rho)
        ratio = (sigmas[1:]-sigmas[:-1])/(sigma_min-sigma_max)
        logits=ratio.log()
        c=1-logits[-1]
        logits=logits+c
        self.init_vec=logits[:-1]
        # # Random init
        # self.init_vec=torch.randn(self.length, dtype=torch.float64)
        # # Resume previous
        # self.init_vec=torch.tensor([13.9053, 12.4946, 11.6138, 11.3109, 11.3066, 10.4056, 10.1062, 10.0252,
        #  9.9265,  9.4689,  9.3377,  8.4978,  8.4505,  8.5741,  8.2184,  7.8147,
        #  8.0148,  7.6851,  7.5641,  7.4591,  7.0903,  6.9518,  7.0726,  6.6815,
        #  6.3316,  6.2892,  5.9725,  6.0144,  5.7710,  5.5893,  5.3469,  5.2438,
        #  5.1350,  4.9320,  4.6835,  4.5535,  4.2176,  3.8566,  3.6656,  3.3114,
        #  1.0000], dtype=torch.float64)
        self.vec=torch.nn.Parameter(self.init_vec)
    
    def forward(self):
        ph=torch.ones_like(self.init_vec[:1], dtype=torch.float64, device=self.vec.device)
        v = torch.cat([self.vec, ph])
        increment=F.softmax(v, dim=0)
        return increment[:self.length]
    
    def sigmas(self):
        lambdas=self.forward()
        sigmas=torch.cumsum(lambdas, dim=0)*(self.sigma_min-self.sigma_max)+self.sigma_max
        sigmas=torch.cat([torch.ones_like(lambdas[:1])*self.sigma_max, sigmas, torch.ones_like(lambdas[:1])*self.sigma_min])
        return sigmas

def training_loop(
    run_dir             = '.',      # Output directory.
    network_dir         = '.',      # Pre-trained DPM
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg      = 10000,    # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_kimg         = 0,        # Start from the given training progress.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()       # batch per GPU
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total                             # batch_gpu <= batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu      # allow for accumulated size larger than batch size
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size(), 'batch size Error: cannot be allocated evenly to GPUs'    # check total batch size

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    # Load pre-trained DPM and construct sigma network.
    dist.print0(f'Loading network from "{network_dir}"...')
    with dnnlib.util.open_url(network_dir, verbose=True) as f:
        net = pickle.load(f)['ema'].to(device)
    net.eval().requires_grad_(False).to(device)
    lambda_net = dnnlib.util.construct_class_by_name(**network_kwargs)
    lambda_net.train().requires_grad_().to(device)
    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
            misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)
    assert images.dtype==torch.float32, f'training dype should be {images.dtype}'

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM)Loss
    optimizer = dnnlib.util.construct_class_by_name(params=lambda_net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    # ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False)
    # ddp = torch.nn.parallel.DistributedDataParallel(lambdas, device_ids=[device], broadcast_buffers=False)
    ema = copy.deepcopy(lambda_net).eval().requires_grad_(False)

    # Resume training from previous snapshot.
    # if resume_pkl is not None:
    #     dist.print0(f'Loading network weights from "{resume_pkl}"...')
    #     if dist.get_rank() != 0:
    #         torch.distributed.barrier() # rank 0 goes first
    #     with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
    #         data = pickle.load(f)
    #     if dist.get_rank() == 0:
    #         torch.distributed.barrier() # other ranks follow
    #     misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
    #     misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
    #     del data # conserve memory
    # if resume_state_dump:
    #     dist.print0(f'Loading training state from "{resume_state_dump}"...')
    #     data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
    #     misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
    #     optimizer.load_state_dict(data['optimizer_state'])
    #     del data # conserve memory

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintain_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    while True:

        # Accumulate gradients.
        optimizer.zero_grad()
        total_loss=0
        total_regu=0
        for round_idx in range(num_accumulation_rounds):
        #     with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
            images, labels = next(dataset_iterator)
            images = images.to(device).to(torch.float32) / 127.5 - 1
            labels = labels.to(device)
            loss = loss_fn(lambda_net=lambda_net, diffusion_net=net, images=images, labels=labels, augment_pipe=None)
            if len(loss)==1:
                loss=loss[0]
            else:
                regu=loss[1].sum().mul(loss_scaling / batch_gpu_total)
                loss=loss[0]
            training_stats.report('Loss/loss', loss)
            loss=loss.sum().mul(loss_scaling / batch_gpu_total)
            total_loss+=loss
            total_regu+=regu
            loss.backward()

        # # Update weights.
        # for g in optimizer.param_groups:
            # g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
            # print(g['lr'])
        for param in lambda_net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()

        # # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), lambda_net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintain_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))
        with torch.no_grad():
            sigmas=lambda_net.sigmas()
        sigmas=[f'{s:.3f}'for s in sigmas]
        dist.print0(f'loss={total_loss:.2f} | regu={total_regu:.2f} | sigmas={sigmas}')

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            # data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            # for key, value in data.items():
            #     if isinstance(value, torch.nn.Module):
            #         value = copy.deepcopy(value).eval().requires_grad_(False)
            #         misc.check_ddp_consistency(value)
            #         data[key] = value.cpu()
            #     del value # conserve memory
            if dist.get_rank() == 0:
                sigmas=ema.sigmas()
                with open(os.path.join(run_dir, 'sigmas-snapshot.txt'), 'a') as f:
                    f.write(f'{cur_nimg//1000:06d} sigmas: ')
                    f.write(str(sigmas))
                    f.write('\n')
            # del data # conserve memory

        # Update logs.
        # training_stats.default_collector.update()
        # if dist.get_rank() == 0:
        #     if stats_jsonl is None:
        #         stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
        #     stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
        #     stats_jsonl.flush()
        # dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintain_time = tick_start_time - tick_end_time
        if done:
            # # Save full dump of the training state.
            if dist.get_rank() == 0:
                torch.save(dict(net=lambda_net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state.pt'))
            # with open(os.path.join(run_dir, 'sigmas-snapshot.txt'), 'a') as f:
            #     for p in ema.parameters():
            #         f.write(str(p))
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------
