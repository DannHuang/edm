"""Loss functions used in the paper
"<placeholder>"."""

import torch
from torch_utils import persistence

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# lambdas define the sigma ratio, lambdas[i]=sigmas[i]/sigmas[i+1], sigmas[T]=sigma_max
# Total DM length is lambdas length+2 (sigma_min and sigma_max).
# 

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.002, sigma_max=80.0):
        self.sigma_min = torch.tensor(sigma_min, dtype=torch.float32)
        self.sigma_max = torch.tensor(sigma_max, dtype=torch.float32)

    def __call__(self, lambda_net, diffusion_net, images, labels, augment_pipe=None):
        lambdas=lambda_net()
        ratio=torch.cat([torch.ones_like(lambdas[:1])*self.sigma_max, lambdas])
        sigmas=torch.cumprod(ratio, dim=0)
        # FIXME: how to guarantee sigmas[-1] > sigma_min? maybe softmax instead of sigmoid.
        # # in this case the sigma function we be addtivie instead of productive.
        last_ratio=torch.ones_like(lambdas[:1])*self.sigma_min/sigmas[-1]
        weights=torch.cat([lambdas, last_ratio])
        dm_length=lambdas.shape[0]
        batch_size=images.shape[0]
        rnd_index = torch.randint(dm_length+1, [batch_size,1,1,1], device=images.device)
        # reg_index = torch.ones([1,1,1,1], device=images.device, dtype=rnd_index.dtype)*dm_length
        # rnd_index = torch.cat([rnd_index, reg_index], dim=0)
        sigma = sigmas[rnd_index]
        weight = 1/weights[rnd_index]-1
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = diffusion_net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y).pow(2).sum())
        # print(f'loss at {sigma[batch_size-1].item():.3f}={loss[batch_size-1,0,0,0].item():.2f}')
        return loss

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
