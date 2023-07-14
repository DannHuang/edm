# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
from torch.autograd.functional import jvp
import torch.autograd.forward_ad as fwAD

#----------------------------------------------------------------------------
# Original EDM sampler (Algorithm 2). 1000img FID=40.4493

def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, k=0
):

    # # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    A = sigma_max ** (1 / rho)
    B = (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    # # Time step discretization, turn time-steps into sigma-schedule
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (A + step_indices/(num_steps-1)*B) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    mean=torch.zeros_like(x_next)
    num=(3*64*64)**0.5
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # # Increase noise temporarily.
        # gamma = min(S_churn / num_steps, 2**0.5 - 1) if S_min <= t_cur <= S_max else 0
        # gamma = 2**0.5 - 1
        # t_hat = net.round_sigma(t_cur + gamma * t_cur)  # 2**0.5 * t_cur
        # # beta * g(t) = (dt)^0.5 * (sigma^2_t)'^0.5
        # x_hat = x_cur + ((t_cur**(1/rho)-t_hat**(1/rho))/B).sqrt() * (rho*(-B)*(t_cur**((2*rho-1)/rho)+t_hat**((2*rho-1)/rho))).sqrt() * S_noise*randn_like(x_cur)
        # x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise*randn_like(x_cur)
        
        # # Euler step.
        denoised = net(x_cur, t_cur, class_labels).to(torch.float64)
        # d_cur = (x_cur - denoised) / t_cur
        # t_hat = d_cur.pow(2).sum().sqrt()/num*t_next
        # rate = t_hat/t_cur
        # noise = randn_like(t_next*d_cur, t_hat, t_cur) if rate>=0.5 else 0
        # x_next = denoised + t_next*d_cur + noise

        rate = t_next/t_cur
        if i==0:
            d_0 = (x_cur - denoised) / t_cur
            x_next = denoised + t_next*d_0
            mean=denoised
        else:
            mean = rate*mean + (1-rate)*denoised
            x_next = mean + t_next*d_0
        # if t_next/t_cur >= 1/2:
        #     noise = randn_like(d_0, t_next, t_cur)
        #     x_next = x_next + noise
        #     d_0 = (t_next*d_0 + noise)/t_next

        # # Apply 2nd order correction.
        # if i < num_steps - 1:
        #     denoised = net(x_next, t_next, class_labels).to(torch.float64)
        #     d_prime = (x_next - denoised) / t_next
        #     d_prime = (t_next*d_prime+t_hat*d_cur)/(t_next+t_hat)
        #     x_next = x_hat + d_prime*(t_next - t_hat)

        # # Logger
        if 0< i < num_steps-1:
            d1 = (center_last - denoised)/t_cur
            l1 = d1.pow(2).sum().sqrt()
            d2 = d_0
            l2 = d2.pow(2).sum().sqrt()
            print(f'{i} | sigma_next: {t_next:.2f} | sigma_hat: {t_cur:.2f} \
                  | inf-norm: {((d2-d1).pow(2).max().sqrt()).item():.2f} | cosine: {((d1*d2).sum()/l1/l2).item():.6f} \
                    | l1: {l1.item():.2f} | l2: {l2.item():.2f}')
        # d_last = d_cur
        center_last=mean

    return x_next

#----------------------------------------------------------------------------
# Our new 1st-order sampler.

def edm_sampler_(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    '''
    Recurrent diffusion
    '''

    # # Hyperparameters:
    linear_steps = 3
    search_steps = 10
    random_init = 1

    # # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_med = min(S_max, sigma_max)
    sigma_max = min(sigma_max, net.sigma_max)
    A = sigma_med ** (1 / rho)
    B = sigma_min ** (1 / rho) - sigma_med ** (1 / rho)

    # # First linear steps / random search steps
    search_steps = list([i*search_steps for i in range(1, linear_steps+1)])
    search_steps = [5, 10, 15]
    ode_step_indices = torch.arange(linear_steps, dtype=torch.float64, device=latents.device)
    ode_sigmas = (sigma_max**(1/rho) + ode_step_indices/(linear_steps)*(sigma_med**(1/rho)-sigma_max**(1/rho)))**rho
    ode_sigmas = torch.tensor([80, 20, 5], dtype=torch.float64, device=latents.device)

    # # Time step discretization, turn time-steps into sigma-schedule
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    sigmas = (A + step_indices/(num_steps-1)*B) ** rho
    sigmas = torch.cat([net.round_sigma(ode_sigmas), net.round_sigma(sigmas), torch.zeros_like(sigmas[:1])]) # t_N = 0
    # print(sigmas)
    # exit()

    # # Main sampling loop.
    flag = False
    x_next = latents.to(torch.float64) * sigmas[0]
    mean = torch.zeros_like(x_next)
    for round in range(random_init):
        # # Time step discretization, turn time-steps into sigma-schedule
        # step_indices = torch.arange(num_steps-round, dtype=torch.float64, device=latents.device)
        # sigmas = (A + step_indices/(num_steps-round-1)*B) ** rho
        # sigmas = torch.cat([net.round_sigma(ode_sigmas), net.round_sigma(sigmas), torch.zeros_like(sigmas[:1])]) # t_N = 0
        if round > 0:
            flag = True
        for i, (sigma_cur, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])): # 0, ..., N-1

            '''
            Taylor expansion of drift term can help compute diffusion step-size precisely.
            Apply when dt is of interest.
            '''
            # dt = (sigma_next**(1/rho) - sigma_cur**(1/rho))/B
            # gamma=torch.tensor(0.0, dtype=torch.float64, device=latents.device)
            # prod=torch.tensor(1.0, dtype=torch.float64, device=latents.device)
            # for j in range(1, int(rho)+1):
            #     prod *= (B*(rho-j+1)/j)
            #     gamma += dt**(j)*prod*sigma_cur**((rho-j)/rho)

            '''
            Random init: start with aiming at denoised return from last epoch
            '''
            # if flag:
            #     diffusion_coff = 2*sigma_med
            #     noise = randn_like(x_next)
            #     x_next = x_next + diffusion_coff * noise
            #     d_cur = (x_next - denoised) /diffusion_coff
            #     x_next = x_next + (sigma_med-diffusion_coff)*d_cur
            #     flag = False
            #     continue

            '''
            # # Apply same Euler step in phase 1: Empirically no performance gaining.
            # if 0<i<linear_steps:
            #     d_cur = (x_next - denoised) / sigma_cur
            #     x_next = x_next + (sigma_next-sigma_cur)*d_cur
            #     continue
            '''

            # # Euler step.
            denoised = net(x_next, sigma_cur, class_labels).to(torch.float64)
            # d_cur = (x_next - denoised) / sigma_cur
            # x_next = x_next + (sigma_next-sigma_cur)*d_cur
            rate = sigma_next/sigma_cur
            if i==0:
                d_0 = (x_next - denoised) / sigma_cur
                x_next = denoised + sigma_next*d_0
                mean=denoised
            else:
                mean = rate*mean + (1-rate)*denoised
                x_next = mean + sigma_next*d_0

            # # Brownian motion
            # # dt.pow(3)*2*(-B).pow(3)*t_cur**((2*rho-3)/rho)+dt.pow(4)*6*(-B).pow(4)*t_cur**((2*rho-4)/rho)+dt.pow(5)*6*(-B).pow(5)*t_cur**((2*rho-5)/rho)+dt.pow(6)*(-B).pow(6)
            # # diffusion_coff = 2 * dt**((1+rho)/2)*((-B)**(rho)*sigma_cur).sqrt()

            if i<linear_steps :
                # l1=d_cur.pow(2).sum().sqrt()
                d_best = (d_0 + (mean-denoised)/sigma_next).pow(2).sum().sqrt()
                x_cur = x_next
                diffusion_coff = 2**0.5-1
                sigma_cur = (1+diffusion_coff)*sigma_next
                rate = sigma_next/sigma_cur
                for _ in range(search_steps[i]):
                    noise = randn_like(x_next)
                    x_hat = x_cur + (diffusion_coff**2 + 2*diffusion_coff)**0.5*sigma_next * noise
                    denoised = net(x_hat, sigma_cur, class_labels).to(torch.float64)
                    d_search = (x_hat - denoised) / sigma_cur
                    if ((mean-denoised)/sigma_next).pow(2).sum().sqrt() < d_best:
                        # print(((mean-denoised)/sigma_next).pow(2).sum().sqrt())
                        d_best=((mean-denoised)/sigma_next).pow(2).sum().sqrt()
                        mean = rate*mean + (1-rate)*denoised
                        x_cur = x_hat + (sigma_next-sigma_cur)*d_search
                x_next = x_cur
                d_0 = (x_next - mean) / sigma_next

            # # Logger
            # if i > 0:
            #     d1=d_cur
            #     d2=d_last
            #     l1 = d1.pow(2).sum().sqrt()
            #     l2 = d2.pow(2).sum().sqrt()
            #     print(f'sigma: {sigma_cur} | gap: {(d1-d2).pow(2).max().item():.2f} | cosine: {((d1*d2).sum()/l1/l2).item():.6f} | l1: {l1.item():.2f} | l2: {l2.item():.2f} | d1 saturation: {(d1.abs()>1e-2).sum()/d1.shape[1]*100:.2f}% | d2 saturation: {(d2.abs()>1e-2).sum()/d2.shape[1]*100:.2f}%')
            # d_last = d_cur
        
            # # Second order correction
            # if linear_steps < i < linear_steps+num_steps - 1:
            #     denoised = net(x_next, sigma_next, class_labels).to(torch.float64)
            #     d_prime = (x_next - denoised) / sigma_next
            #     d_prime = (sigma_next*d_prime+sigma_cur*d_cur)/(sigma_cur+sigma_next)
            #     x_next = x_cur + (sigma_next-sigma_cur)*d_prime

    return x_next

'''
#----------------------------------------------------------------------------
# Our new 2nd-order sampler. 1000img FID = 41.5904

def edm_sampler_(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, k=0
):

    # # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    A = torch.tensor(sigma_max**(1/rho), dtype=torch.float64)
    B = torch.tensor(sigma_min**(1/rho) - sigma_max**(1/rho), dtype=torch.float64)

    # # Time step discretization, turn time-steps into sigma-schedule.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)  # [0,1,...,num_steps-1]
    sigmas = (A + step_indices/(num_steps-1)*B).pow(rho)
    sigmas = torch.cat([net.round_sigma(sigmas), torch.zeros_like(sigmas[:1])]) # t_steps[num_steps] = 0
    dt = torch.tensor(1/(num_steps-1), dtype=torch.float64, device=latents.device)

    # # Main sampling loop.
    x_next = latents.to(torch.float64) * sigmas[0]     # amplify to sigma_max variance
    for i, (sigma_cur, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])): # 0, ..., N-1
        x_cur = x_next
        dt = (sigma_next**(1/rho) - sigma_cur**(1/rho))/B      # dt>0

        # # increase nosie level except last iteration
        if i<num_steps-1:
            diffusion_coff = dt**((rho)/2)*((-B)**(rho)*sigma_cur).sqrt()
            if randn_like.__name__ == 'multiGaussian_like':
                noise = randn_like(x_cur, dt)
            else:
                noise = [dt**0.5*randn_like(x_cur)]
            x_cur = x_cur + diffusion_coff * noise[0]
            sigma_cur = (sigma_cur**2 + dt*diffusion_coff**2).sqrt()
            dt = (sigma_next**(1/rho) - sigma_cur**(1/rho))/B

        gamma=torch.tensor(0.0, dtype=torch.float64, device=latents.device)
        prod=torch.tensor(1.0, dtype=torch.float64, device=latents.device)
        for j in range(1, int(rho)):
            prod *= (B*(rho-j+1)/j)
            gamma += dt**(j-1)*prod*sigma_cur**((rho-j)/rho)

        # # backward 2nd order sampling.
        if i > 0 and sigma_cur >= S_max:
            with fwAD.dual_level():
                dual_x = fwAD.make_dual(x_cur, 0.5*dt*dt*gamma/sigma_cur*f_cur+noise[1]/sigma_cur)
                dual_t = fwAD.make_dual(sigma_cur, 0.5*dt*dt/sigma_cur)
                dual_out = net(dual_x, dual_t, class_labels)
                denoised, jfp = fwAD.unpack_dual(dual_out)
            jfp = (dt*dt*gamma/sigma_cur*f_cur + 0.5*dt*dt/sigma_cur*diffusion_coff*noise[0] + noise[1]/sigma_cur -jfp).to(torch.float64)    # 0.5*dt^2 * (x_next-x_cur)
            f_cur = (x_cur - denoised) / sigma_cur
            x_next = x_cur + (f_cur*dt + jfp)*gamma
        else:
            # # Euler step.
            denoised = net(x_cur, sigma_cur, class_labels).to(torch.float64)
            f_cur = (x_cur - denoised) / sigma_cur
            # # v1.0 implementation
            # dsigma = B*rho*(sigma_cur.pow((rho-1)/rho))
            # ddsigma = rho*(rho-1)*B*B*(sigma_cur.pow((rho-2)/rho))
            # g_cur = ddsigma*sigma_cur
            # gamma = dsigma + 0.5*dt*ddsigma     
            x_next = x_next + f_cur*gamma*dt
        
        # if sigma_cur < S_min and i < num_steps - 1:
        #     denoised = net(x_next, sigma_next, class_labels).to(torch.float64)
        #     # gamma_next=torch.tensor(0.0, dtype=torch.float64, device=latents.device)
        #     # prod=torch.tensor(1.0, dtype=torch.float64, device=latents.device)
        #     # for j in range(1, int(rho)):
        #     #     prod *= (B*(rho-j+1)/j)
        #     #     gamma_next += dt**(j-1)*prod*sigma_next**((rho-j)/rho)
        #     f_prime = (x_next - denoised) / sigma_next
        #     x_next = x_next + 0.5*(f_prime*gamma-f_cur*gamma)*dt

    return x_next
    '''

#----------------------------------------------------------------------------
# Generalized ablation sampler, representing the superset of all sampling
# methods discussed in the paper.

def ablation_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='heun', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == 'heun'
            denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels).to(torch.float64)
            d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

    return x_next

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])
    
    def multiGaussian_like(self, input_tensor, d_t, **kwargs):
        '''
        return d_beta and 2nd-order d_beta
        '''
        size = input_tensor.shape
        device = input_tensor.device
        assert size[0] == len(self.generators)
        # Cholesky decomposition of covariance matrix
        up_left = ((1/3)*d_t**3)**0.5
        bottom_left = 0.5*(3*d_t)**0.5
        bottom_right = 0.5*d_t**0.5
        eps = [torch.randn([2, size[1], size[2], size[3]], generator=gen, **kwargs, device=device) for gen in self.generators]
        dd_beta = torch.stack([noise[0]*up_left for noise in eps])
        d_beta = torch.stack([noise[0]*bottom_left + noise[1]*bottom_right for noise in eps])
        return (d_beta, dd_beta)

    def corrGaussian_like(self, input_tensor, input_var, output_var, **kwargs):
        '''
        Return noise correlated with input tensor
        param: input_tensor: Gaussian with std input_var
        param: input_var: std of input tensor
        param: output_var: std of output tensor
        '''
        size = input_tensor.shape
        device = input_tensor.device
        assert size[0] == len(self.generators)
        # Cholesky decomposition of covariance matrix
        bottom_left = -output_var**2/input_var**2/2
        bottom_right = output_var*(1-output_var**2/input_var**2/4)**0.5
        eps = [torch.randn([size[1], size[2], size[3]], generator=gen, **kwargs, device=device) for gen in self.generators]
        # dd_beta = torch.stack([input_tensor for _ in range(size[0])])
        d_beta = torch.stack([noise*bottom_right for noise in eps])
        d_beta = input_tensor*bottom_left + d_beta
        return d_beta

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])
    
def multiGaussian_like(input_tensor, d_t, **kwargs):
    '''
    return d_beta and 2nd-order d_beta
    '''
    size = input_tensor.shape
    device = input_tensor.device
    # Cholesky decomposition of covariance matrix
    up_left = ((1/3)*d_t**3)**0.5
    bottom_left = 0.5*(3*d_t)**0.5
    bottom_right = 0.5*d_t**0.5
    # up_right = torch.zeros_like(up_left)
    # left = torch.concat((up_left, bottom_left) , dim=0)
    # right = torch.concat((up_right, bottom_right) , dim=0)
    # mean = torch.concat((left, right), dim=1)
    eps = torch.randn([2, size[1], size[2], size[3]], **kwargs, device=device)
    dd_beta = eps[0]*up_left
    d_beta = eps[0]*bottom_left + eps[1]*bottom_right
    return (d_beta, dd_beta)

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)
# sampler_kwargs
@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)
# @click.option('--k',                       help='residual order of diffusion-coefficient', metavar='FLOAT',         type=click.FloatRange(min=0, min_open=False), default=0, show_default=True)
@click.option('--randn_like',              help='Stoch. Brownian motions generator', metavar='db|ddb',              type=click.Choice(['db', 'ddb']), default='db')

# ablation
@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))

def main(network_pkl, outdir, seeds, subdirs, class_idx, max_batch_size, device=torch.device('cuda'), **sampler_kwargs):

    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    if network_pkl.startswith('https'):
        dist.print0(f'Loading network from url "{network_pkl}"...')
        with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
            net = pickle.load(f)['ema'].to(device)
    else:
        dist.print0(f'Loading network from local directorty "{network_pkl}"...')
        with open(network_pkl, 'rb') as f:
            net = pickle.load(f).to(device)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]        # random initialize class vector
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1      # one-hot

        # Generate images.
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}     # unwrap kwargs, withdraw non-stated params
        have_ablation_kwargs = any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
        if 'randn_like' in sampler_kwargs and type(sampler_kwargs['randn_like']) is str:
            sampler_kwargs['randn_like'] = rnd.randn_like if sampler_kwargs['randn_like'] == 'db' else rnd.corrGaussian_like
        sampler_fn = ablation_sampler if have_ablation_kwargs else edm_sampler
        images = sampler_fn(net, latents, class_labels, **sampler_kwargs)

        # Save images.
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        
        # # Save batch images
        # images_dir = os.path.join(outdir, f'{class_idx:02d}') if subdirs else outdir
        # os.makedirs(images_dir, exist_ok=True)
        # images_path = os.path.join(images_dir, f'{class_idx:02d}.png')
        # PIL.Image.fromarray(images_np, 'RGB').save(images_path)
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
            else:
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)

    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
