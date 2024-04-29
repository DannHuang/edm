import dnnlib
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import misc
import torch_utils.distributed as dist
from generate import StackedRandomGenerator

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
    def __init__(self, diffusion_length=10, sigma_max=80.0, sigma_min=0.002, rho=3.0):
        super().__init__()
        self.length=diffusion_length - 2
        self.sigma_max=sigma_max
        self.sigma_min=sigma_min
        self.rho=rho
        self.init_shift = torch.tensor(self.sigma_max**(1/self.rho), dtype=torch.float64)
        self.init_scale = torch.tensor(self.sigma_min**(1/self.rho) - self.sigma_max**(1/self.rho), dtype=torch.float64)
        step_indices = torch.arange(diffusion_length, dtype=torch.float64)  # [0,1,...,num_steps-1]
        sigmas = (self.init_shift + step_indices/(diffusion_length - 1) * self.init_scale).pow(self.rho)
        ratio = (sigmas[1:]-sigmas[:-1]) / (sigma_min-sigma_max)
        logits=ratio.log()
        c=1-logits[-1]
        logits=logits+c
        self.init_vec=logits[:-1]
        # # Random init
        # self.init_vec=torch.randn(self.length, dtype=torch.float64)
        self.vec=torch.nn.Parameter(self.init_vec, requires_grad=True)

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

class softmax_model_batch(nn.Module):
    def __init__(self, diffusion_length = 10, sigma_max = 80.0, sigma_min = 0.002, rho = 3.0, init='edm'):
        super().__init__()
        self.dm_length = diffusion_length      # sigma min/max is fixed, loss 2 degrees of freedom
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        if init == 'edm':
            self.rho = rho
            self.init_shift = torch.tensor(self.sigma_max**(1/self.rho))
            self.init_scale = torch.tensor(self.sigma_min**(1/self.rho) - self.sigma_max**(1/self.rho))
            step_indices = torch.arange(diffusion_length)    # [0,1,...,num_steps-1]
            sigmas = (self.init_shift + step_indices/(diffusion_length - 1) * self.init_scale).pow(self.rho)
            increments = (sigmas[1:]-sigmas[:-1]) / (sigma_min-sigma_max)           # dm_length - 1
            logits = increments.log()
            c = 1 - logits[-1]
            logits = logits + c
            self.init_vec = logits[:-1]             # less 1 degree of freedom due to soft-max. dm_length - 2
        else:
            self.init_vec=torch.randn(self.dm_length - 2)
        self.vec = torch.nn.Parameter(self.init_vec, requires_grad = True)

    def forward(self, positions):
        ph = torch.ones_like(self.init_vec[:1], device=self.vec.device)
        v = torch.cat([self.vec, ph])
        increment = F.softmax(v, dim = 0)
        percentage = torch.einsum('i,bji->bj', increment, positions)
        sigmas = percentage * (self.sigma_min - self.sigma_max) + self.sigma_max
        return sigmas

class finetune_wrapper(nn.Module):

    def __init__(
            self,
            diffusion_network_kwargs,
            interface_kwargs,
            sigma_model_kwargs,
            loss_mode='Dns',
            pretrained_dm=None,
            ):

        super().__init__()
        d_model = dnnlib.util.construct_class_by_name(**diffusion_network_kwargs, **interface_kwargs)
        self.diffusion = d_model.train()
        s_model = dnnlib.util.construct_class_by_name(**sigma_model_kwargs)
        self.sigma_model = s_model.train()
        self.loss_mode = loss_mode
        if pretrained_dm is not None:
            self.init_from_pretrained(pretrained_dm)

    def init_from_pretrained(self, pretrained_dm):
        dist.print0(f'Loading network weights from "{pretrained_dm}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(pretrained_dm, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=self.diffusion, require_all=False)
        del data # conserve memory

    def _get_loss_legacy(self, images, labels, augment_pipe=None):
        torch.cuda.reset_peak_memory_stats()
        batch_size = images.shape[0]
        t = np.random.randint(self.sigma_model.dm_length - 1, size=batch_size)   # length-1 increments
        index = [[j for j in range(i)] for i in t]
        summation_vec = np.zeros([batch_size, self.sigma_model.dm_length - 1])
        index_next = [[j for j in range(i)] for i in t+1]
        summation_vec_next = np.zeros([batch_size, self.sigma_model.dm_length - 1])
        for i in range(batch_size):
            summation_vec[i, index[i]] = 1
            summation_vec_next[i, index_next[i]] = 1

        summation_tensor = torch.stack((torch.from_numpy(summation_vec), torch.from_numpy(summation_vec_next)), dim=1)
        print('before forward:', torch.cuda.max_memory_allocated(images.device)/ 2**30)
        sigmas = self.sigma_model(summation_tensor.to(images.device)).unsqueeze(-1).unsqueeze(-1)  # batch of [cur_sigma, next_sigma]
        cur_sigma, next_sigma = sigmas.chunk(2, dim = 1)    # [batch, 1]
        if self.loss_mode == 'Dns':
            # denoised weights
            weights = 1 / next_sigma - 1 / cur_sigma
        else:
            # epsilon weights
            weights = next_sigma / cur_sigma
            weights = 1 / weights - 1

        # reg_index = torch.ones([1,1,1,1], device = images.device, dtype = rnd_index.dtype) * dm_length
        # rnd_index = torch.cat([rnd_index, reg_index], dim=0)
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * cur_sigma+next_sigma
        D_yn = self.diffusion(y + n, cur_sigma+next_sigma, labels, augment_labels=augment_labels)
        loss = weights * ((D_yn - y).pow(2))
        print('after forward:', torch.cuda.max_memory_allocated(images.device)/ 2**30)
        return loss

    def forward(self, images, labels, summation_tensor, augment_pipe=None):
        sigmas = self.sigma_model(summation_tensor.to(images.device)).unsqueeze(-1).unsqueeze(-1)  # batch of [cur_sigma, next_sigma]
        cur_sigma, next_sigma = sigmas.chunk(2, dim = 1)    # [batch, 1]
        if self.loss_mode == 'Dns':
            # denoised weights
            weights = 1 / next_sigma - 1 / cur_sigma
        else:
            # epsilon weights
            weights = next_sigma / cur_sigma
            weights = 1 / weights - 1

        # reg_index = torch.ones([1,1,1,1], device = images.device) * dm_length
        # rnd_index = torch.cat([rnd_index, reg_index], dim=0)
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * cur_sigma
        D_yn = self.diffusion(y + n, cur_sigma, labels, augment_labels=augment_labels)
        loss = weights * ((D_yn - y).pow(2))
        return loss

    def get_loss(self, images, labels, augment_pipe=None):
        batch_size = images.shape[0]
        t = np.random.randint(self.sigma_model.dm_length - 1, size=batch_size)   # length-1 increments
        index = [[j for j in range(i)] for i in t]
        summation_vec = np.zeros([batch_size, self.sigma_model.dm_length - 1])
        index_next = [[j for j in range(i)] for i in t+1]
        summation_vec_next = np.zeros([batch_size, self.sigma_model.dm_length - 1])
        for i in range(batch_size):
            summation_vec[i, index[i]] = 1
            summation_vec_next[i, index_next[i]] = 1

        summation_tensor = torch.stack((torch.from_numpy(summation_vec), torch.from_numpy(summation_vec_next)), dim=1).to(torch.float32)
        return self(images, labels, summation_tensor, augment_pipe)

    def sample(self, batch_size=64, class_idx=None, device=None, seeds=None):
        with torch.no_grad():
            # Create learned schedule
            t = np.array([i for i in range(self.sigma_model.dm_length)])
            index = [[j for j in range(i)] for i in t]
            summation_vec = np.zeros([self.sigma_model.dm_length, self.sigma_model.dm_length - 1])
            summation_vec_next = np.zeros([self.sigma_model.dm_length, self.sigma_model.dm_length - 1])
            for i in range(self.sigma_model.dm_length):
                summation_vec[i, index[i]] = 1
            summation_tensor = torch.stack((torch.from_numpy(summation_vec), torch.from_numpy(summation_vec_next)), dim=1).to(torch.float32)
            sigma = self.sigma_model(summation_tensor.to(device))
            sigmas, _ = sigma.chunk(2, dim=1)
            # TODO
            # sigmas = torch.cat([self.diffusion.round_sigma(sigmas), torch.zeros_like(sigmas[:1])])
            sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])])
            dist.print0(f'Sampling with sigmas: {[s.item() for s in sigmas.squeeze()]}')

            # Pick latents and labels.
            seeds = [i for i in range(batch_size)] if seeds is None else seeds
            rnd = StackedRandomGenerator(device, seeds)
            latents = rnd.randn([batch_size, self.diffusion.img_channels, self.diffusion.img_resolution, self.diffusion.img_resolution], device=device)
            class_labels = None
            if self.diffusion.label_dim:
                class_labels = torch.eye(self.diffusion.label_dim, device=device)[rnd.randint(self.diffusion.label_dim, size=[batch_size], device=device)]
            if class_idx is not None:
                class_labels[:, :] = 0
                class_labels[:, class_idx] = 1      # one-hot

            # Sampling loop.
            x_next = latents.to(torch.float64) * sigmas[0]     # amplify to sigma_max variance
            for i, (sigma_cur, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])): # 0, ..., N-1
                x_cur = x_next
                # # Euler step.
                denoised = self.diffusion(x_cur, sigma_cur, class_labels).to(torch.float64)
                eps = (x_cur - denoised) / sigma_cur
                x_next = denoised + sigma_next * eps

        return x_next