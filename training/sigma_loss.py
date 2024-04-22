"""Loss functions used in the paper
"<placeholder>"."""

import numpy as np
import torch
from torch_utils import persistence

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class SoftmaxLoss:

    def __init__(self, mode='Dns'):
        self.mode=mode

    def __call__(self, sigma_model, diffusion_net, images, labels, augment_pipe=None):
        batch_size = images.shape[0]
        t = np.random.randint(sigma_model.dm_length - 1, size=batch_size)   # length-1 increments
        index = [[j for j in range(i)] for i in t]
        summation_vec = np.zeros([batch_size, sigma_model.dm_length - 1])
        index_next = [[j for j in range(i)] for i in t+1]
        summation_vec_next = np.zeros([batch_size, sigma_model.dm_length - 1])
        for i in range(batch_size):
            summation_vec[i, index[i]] = 1
            summation_vec_next[i, index_next[i]] = 1

        summation_tensor = torch.stack((torch.from_numpy(summation_vec), torch.from_numpy(summation_vec_next)), dim=1)
        sigmas = sigma_model(summation_tensor.to(images.device))  # batch of [cur_sigma, next_sigma]
        cur_sigma, next_sigma = sigmas.chunk(2, dim = 1)    # [batch, 1]
        print(cur_sigma, next_sigma)
        if self.mode == 'Dns':
            # denoised weights
            weights = 1 / next_sigma - 1 / cur_sigma
        else:
            # epsilon weights
            weights = next_sigma / cur_sigma
            weights = 1 / weights - 1

        # reg_index = torch.ones([1,1,1,1], device = images.device, dtype = rnd_index.dtype) * dm_length
        # rnd_index = torch.cat([rnd_index, reg_index], dim=0)
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * cur_sigma
        D_yn = diffusion_net(y + n, cur_sigma, labels, augment_labels=augment_labels)
        loss = weights * ((D_yn - y).pow(2))
        return loss

#----------------------------------------------------------------------------
# lambdas define the sigma ratio, lambdas[i]=sigmas[i]/sigmas[i+1], sigmas[T]=sigma_max
# Total DM length is lambdas length+2 (sigma_min and sigma_max).
# 

@persistence.persistent_class
class SigmoidLoss:
    def __init__(self, sigma_min=0.002, sigma_max=80.0, mode='Dns'):
        self.sigma_min = torch.tensor(sigma_min, dtype=torch.float64)
        self.sigma_max = torch.tensor(sigma_max, dtype=torch.float64)
        self.mode = mode

    def __call__(self, lambda_net, diffusion_net, images, labels, augment_pipe=None):
        lambdas=lambda_net()
        ratio=torch.cat([torch.ones_like(lambdas[:1])*self.sigma_max, lambdas])
        sigmas=torch.cumprod(ratio, dim=0)
        dm_length=lambdas.shape[0]       # lambda length + 1
        batch_size=images.shape[0]
        if self.mode=='Eps':
            # epsilon weights
            last_ratio=torch.ones_like(lambdas[:1])*self.sigma_min/sigmas[-1]
            weights=torch.cat([lambdas, last_ratio])
            weights=1/weights-1
        else:
            # denoised weights
            # last_ratio=torch.ones_like(lambdas[:1])*self.sigma_min/sigmas[-1]
            # weights=torch.cat([lambdas, torch.zeros_like(lambdas[:1])])
            weights=1-lambdas
        rnd_index = torch.randint(dm_length, [batch_size,1,1,1], device=images.device)
        sigma = sigmas[rnd_index]
        weight = weights[rnd_index]
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        with torch.no_grad():
            D_yn = diffusion_net(y + n, sigma, labels, augment_labels=augment_labels)
        n_last = torch.randn_like(y) * sigmas[-1]
        D_yn_last = diffusion_net(y + n_last, sigmas[-1], labels, augment_labels=augment_labels)
        regu = ((D_yn_last - y).pow(2))/sigmas[-1]
        # weight_loss = weight * ((D_yn - y).pow(2))
        loss = weight * ((D_yn - y).pow(2)) + regu
        # print(f'loss at each step: {((D_yn - y).pow(2).sum()/batch_size).tolist()}')
        # print(f'loss at {sigma[batch_size-1].item():.3f}={loss[batch_size-1,0,0,0].item():.2f}')
        return (loss, regu)

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
