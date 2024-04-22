# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

import os
import torch
from . import training_stats

#----------------------------------------------------------------------------

def init():
    if 'MASTER_ADDR' not in os.environ:
        print(f"Set MASTER_ADDR to 'localhost'.")
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        print(f"Set MASTER_PORT to '29500'.")
        os.environ['MASTER_PORT'] = '29500'
    if 'RANK' not in os.environ:
        print(f"Set RANK to '0'.")
        os.environ['RANK'] = '0'
    if 'LOCAL_RANK' not in os.environ:
        print(f"Set LOCAL_RANK to '0'.")
        os.environ['LOCAL_RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        print(f"Set WORLD_SIZE to '1'.")
        os.environ['WORLD_SIZE'] = '1'

    backend = 'gloo' if os.name == 'nt' else 'nccl'
    print(f"Rank-{os.environ['RANK']} initializing ...")
    torch.distributed.init_process_group(backend=backend, init_method='env://')
    print0(f"torch distributed initialized with {get_world_size()} processes.")
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))

    sync_device = torch.device('cuda') if get_world_size() > 1 else None
    training_stats.init_multiprocessing(rank=get_rank(), sync_device=sync_device)

#----------------------------------------------------------------------------

def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

#----------------------------------------------------------------------------

def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

#----------------------------------------------------------------------------

def should_stop():
    return False

#----------------------------------------------------------------------------

def update_progress(cur, total):
    _ = cur, total

#----------------------------------------------------------------------------

def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)

#----------------------------------------------------------------------------
