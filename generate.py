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

#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).

def edm_sampler_(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    A = sigma_max ** (1 / rho)
    B = (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    # Time step discretization, turn time-steps into sigma-schedule
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (A + step_indices/(num_steps-1)*B) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    
    # # approximation of (t_next - t_cur)
    # d_sigma = []
    # d_t_sigma = []
    # d_t = []
    # for _, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1

    #     # gamma = min(S_churn / num_steps, 2**0.5 - 1) if S_min <= t_cur <= S_max else 0
    #     # t_hat = net.round_sigma(t_cur + gamma * t_cur)
    #     # print(f'gamma={gamma}, dt={(t_next**(1/rho) - t_hat**(1/rho))}')

    #     d_sigma.append((t_next**2))
    #     dt = (t_next**(1/rho) - t_cur**(1/rho))/B
    #     d_t.append(dt)
    #     dtsigma = dt *B*rho * (3/3*t_cur**((rho-1)/rho)+0/3*t_next**((rho-1)/rho)) + 0.5*dt*dt*rho*(rho-1)*B*B * (6/6*t_cur**((rho-2)/rho)+0/12*t_next**((rho-2)/rho))
    #     d_t_sigma.append((t_cur + dtsigma)**2)
    #     # d_t_sigma.append(dt *B*rho * (t_cur**((rho-1)/rho)+t_next**((rho-1)/rho))/2)
    # d_t_sigma = torch.tensor(d_t_sigma, device='cuda')
    # d_sigma = torch.tensor(d_sigma, device='cuda')
    # d_t = torch.tensor(d_t, device='cuda')
    # # print(d_t)
    # print(((d_t_sigma-d_sigma)**2).sum().sqrt())
    # exit()

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, 2**0.5 - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        if randn_like.__name__ == 'multiGaussian_like':
            correlated_noise = randn_like(x_cur, (t_cur**(1/rho)-t_hat**(1/rho))/B)
            x_hat = x_cur + (rho*(-B)*(t_cur**((2*rho-1)/rho)+t_hat**((2*rho-1)/rho))).sqrt() * S_noise * correlated_noise[0]
        else:
            # # beta * g(t) = (dt)^0.5 * (sigma^2_t)'^0.5
            # x_hat = x_cur + ((t_cur**(1/rho)-t_hat**(1/rho))/B).sqrt() * (rho*(-B)*(t_cur**((2*rho-1)/rho)+t_hat**((2*rho-1)/rho))).sqrt() * S_noise*randn_like(x_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise*randn_like(x_cur)
        
        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        # _, Jf=jvp(net, (x_hat, t_hat, class_labels), (d_cur,torch.zeros_like(t_cur),torch.zeros_like(class_labels)))
        # Jf = ((d_cur-Jf) /t_hat *2*rho*B/(x_hat)**(1/rho)).to(torch.float64)
        
        # # dx = -eps_t * B*dt * 
        dt = (t_next**(1/rho) - t_hat**(1/rho))/B
        # x_next = x_hat + d_cur * dt * (B*rho*t_hat**((rho-1)/rho) + 0.5*rho*(rho-1)*B*B*dt * t_hat**((rho-2)/rho))
        # x_next = x_hat + d_cur * dt * B*rho * ((t_hat**((rho-1)/rho)+t_next**((rho-1)/rho)) + 0.5*(rho-1)*B*dt * (t_hat**((rho-2)/rho)+t_next**((rho-2)/rho)))/2
        # # dx = -eps_t * d(sigma_t)
        # print((B*rho* (t_hat**((rho-1)/rho)) * dt + 0.5*rho*(rho-1)*B*B * (t_hat**((rho-2)/rho)) * dt*dt)-(t_next - t_hat))
        x_next = x_hat + d_cur * (t_next - t_hat)       # negative delta-t here (t_next - t_hat)
        
        # 2nd order deriviatives
        # x_next = x_hat + d_cur * (t_next**(1/rho) - t_hat**(1/rho)) * rho * (t_hat**((rho-1)/rho)+t_next**((rho-1)/rho))/2 + Jf*((t_next**(1/rho) - t_hat**(1/rho))/B)**2

        # Apply 2nd order correction.
        # if i < num_steps - 1:
        #     denoised = net(x_next, t_next, class_labels).to(torch.float64)
        #     d_prime = (x_next - denoised) / t_next

        #     _, Jf=jvp(net, (x_next, t_next, class_labels), (d_prime,torch.zeros_like(t_cur),torch.zeros_like(class_labels)))
        #     Jf = ((d_prime-Jf) /t_next ).to(torch.float64)

        #     x_next = x_hat + (0.5 * d_cur + 0.5 * d_prime) * (t_next**(1/rho) - t_hat**(1/rho)) * rho/2 * (t_hat**((rho-1)/rho)+t_next**((rho-1)/rho))
        #     # x_next = x_hat + (0.5 * d_cur + 0.5 * d_prime) * (t_next - t_hat)
        #     # x_next = x_hat + (0.5 * d_cur + 0.5 * d_prime + (t_next**(1/rho) - t_hat**(1/rho)) * rho/2 * (t_hat**((rho-1)/rho)+t_next**((rho-1)/rho))*Jf) * (t_next**(1/rho) - t_hat**(1/rho)) * rho/2 * (t_hat**((rho-1)/rho)+t_next**((rho-1)/rho))
        
        # # backward difference
        # if i > 0:
        #     # _, Jf=jvp(net, (x_next, t_next, class_labels), (d_prime,torch.zeros_like(t_cur),torch.zeros_like(class_labels)))
        #     # Jf = ((d_prime-Jf) /t_next ).to(torch.float64)
        #     # print((Jf**2).sum().sqrt())
        #     # print((d_prime**2).sum().sqrt())

        #     x_next = x_hat + (0.5 * d_cur + 0.5 * d_last) * (t_next**(1/rho) - t_hat**(1/rho)) * rho/2 * (t_hat**((rho-1)/rho)+t_last**((rho-1)/rho))
        #     # x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime + (t_next - t_hat)*Jf)
        # d_last = d_cur
        # t_last = t_hat

    return x_next

def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    A = sigma_max ** (1 / rho)
    B = (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    # Time step discretization, turn time-steps into sigma-schedule
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (A + step_indices/(num_steps-1)*B) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next
        
        # Euler step.
        denoised = net(x_cur, t_cur, class_labels).to(torch.float64)
        d_cur = (x_cur - denoised) / t_cur
        # _, Jf=jvp(net, (x_cur, t_cur, class_labels), (d_cur,torch.zeros_like(t_cur),torch.zeros_like(class_labels)))

        dt = (t_next**(1/rho) - t_cur**(1/rho))/B       # positive dt
        # gamma = dt * (B*rho*(t_cur**((rho-1)/rho)) + 0.5*dt*rho*(rho-1)*B*B*(t_cur**((rho-2)/rho)))
        # Jf = (gamma**2*(d_cur-Jf)/t_cur).to(torch.float64)
        x_next = x_cur + d_cur*dt * (B*rho*(t_cur**((rho-1)/rho)) + 0.5*dt*rho*(rho-1)*B*B*(t_cur**((rho-2)/rho)))
        # # dx = -eps_t * d(sigma_t)
        # x_next = x_hat + d_cur * (t_next - t_hat)       # negative delta-t here (t_next - t_hat)
        
        # 2nd order deriviatives
        # x_next = x_hat + d_cur * (t_next**(1/rho) - t_hat**(1/rho)) * rho * (t_hat**((rho-1)/rho)+t_next**((rho-1)/rho))/2 + Jf*((t_next**(1/rho) - t_hat**(1/rho))/B)**2

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next

            # _, Jf=jvp(net, (x_next, t_next, class_labels), (d_prime,torch.zeros_like(t_cur),torch.zeros_like(class_labels)))
            # Jf = ((d_prime-Jf) /t_next ).to(torch.float64)

            x_next = x_cur + 0.5*(d_cur+d_prime)*dt * (B*rho*(t_cur**((rho-1)/rho)) + 0.5*dt*rho*(rho-1)*B*B*(t_cur**((rho-2)/rho)))
            # x_next = x_cur + (0.5 * d_cur + 0.5 * d_prime) * (t_next - t_cur)
            # x_next = x_hat + (0.5 * d_cur + 0.5 * d_prime + (t_next**(1/rho) - t_hat**(1/rho)) * rho/2 * (t_hat**((rho-1)/rho)+t_next**((rho-1)/rho))*Jf) * (t_next**(1/rho) - t_hat**(1/rho)) * rho/2 * (t_hat**((rho-1)/rho)+t_next**((rho-1)/rho))
        
        # # backward difference
        # if i > 0:
        #     # _, Jf=jvp(net, (x_next, t_next, class_labels), (d_prime,torch.zeros_like(t_cur),torch.zeros_like(class_labels)))
        #     # Jf = ((d_prime-Jf) /t_next ).to(torch.float64)
        #     # print((Jf**2).sum().sqrt())
        #     # print((d_prime**2).sum().sqrt())

        #     x_next = x_hat + (0.5 * d_cur + 0.5 * d_last) * (t_next**(1/rho) - t_hat**(1/rho)) * rho/2 * (t_hat**((rho-1)/rho)+t_last**((rho-1)/rho))
        #     # x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime + (t_next - t_hat)*Jf)
        # d_last = d_cur
        # t_last = t_hat
        
        k=2         # hyper-param, range=[0,]
        if randn_like.__name__ == 'multiGaussian_like':
            correlated_noise = randn_like(x_cur, dt)
            x_next = x_next + (dt*((1+k)/2))*(rho*(rho-1)*B*B*(t_cur**((2*rho-2)/rho))).sqrt() * S_noise*correlated_noise[0]
        else:
            # # beta * g(t) = (dt)^0.5 * (sigma^2_t)'^0.5
            x_next = x_next + (dt**((2+k)/2))*(rho*(rho-1)*B*B*(t_cur**((2*rho-2)/rho))).sqrt() * S_noise*randn_like(x_cur)
            # x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise*randn_like(x_cur)

    return x_next

def edm_sampler_2nd(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    '''
    Input:
            latents: standard Gaussian noise x_T
    '''
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    A = sigma_max ** (1 / rho)
    B = (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    # Time step discretization, turn time-steps into sigma-schedule
    # t_steps = sigma-schedule
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)  # [0,1,...,num_steps-1]
    t_steps = (A + step_indices/(num_steps-1)*B) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_steps[num_steps] = 0
    # # t_steps = [sigma_max, ..., sigma_min, 0]

    # Main sampling loop.
    x_cur = latents.to(torch.float64) * t_steps[0]     # amplify to sigma_max variance
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        
        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        # sample Brownian motion
        noise = randn_like(x_cur, t_hat-t_cur)
        dd_beta, d_beta = noise[1], noise[0]

        # Euler step.
        denoised = net(x_cur, t_cur, class_labels).to(torch.float64)
        d_cur = (x_cur - denoised) * (3*t_cur+t_next) /2 /t_cur**2      # f
        # x_next = x_cur + (t_next-t_cur)*d_cur +(t_next+t_cur).sqrt()*d_beta
        x_next = x_cur + (t_next-t_cur)*d_cur +d_beta

        # Apply 2nd order correction.
        if i > 0:
            _, Jf=jvp(net, (x_cur, t_cur, class_labels), (d_cur,torch.zeros_like(t_cur),torch.zeros_like(class_labels)))
            _, JL=jvp(net, (x_cur, t_cur, class_labels), (dd_beta,torch.zeros_like(t_cur),torch.zeros_like(class_labels)))

            ########## JVP test ##########
            # x_next.requires_grad_()
            # output = net(x_next, t_next, class_labels).to(JacoVP.dtype)
            # res = torch.zeros_like(x_next, dtype=JacoVP.dtype)
            # for i in range(3):
            #     for j in range(32):
            #         for k in range(32):
            #             output[0][i][j][k].backward(retain_graph=True)
            #             res[0][i][j][k]=((x_next.grad)*d_cur).sum()
            #             x_next.grad.zero_()
            # print(torch.allclose(res.to(torch.float32), JacoVP.to(torch.float32)))
            # print(torch.pow(res.to(torch.float32)-JacoVP.to(torch.float32), 2).sum())
            # sys.exit()
            ###############################

            Jf = ((d_cur-Jf) * (3*t_cur+t_next) /2 /t_cur**2).to(torch.float64)
            JL = ((torch.stack([torch.eye(x_cur.shape[-1], device=x_cur.device)]*3)-JL) * ((t_next+t_cur)**0.5) * (3*t_cur+t_next)/2/t_cur**2).to(torch.float64)
            x_next = x_cur + (t_next-t_cur) * ((0.5*d_last + 0.5*d_cur) + (t_next-t_cur) * (Jf)) + d_beta+JL
        d_last = d_cur

    return x_next

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
# sampler
@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)
# ablation
@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))
@click.option('--randn_like', 'randn_like',  help='Stoch. Brownian motions generator', metavar='db|ddb',              type=click.Choice(['db', 'ddb']), default='db')

def main(network_pkl, outdir, subdirs, seeds, class_idx, max_batch_size, device=torch.device('cuda'), **sampler_kwargs):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Generate 64 images and save them as out/*.png
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    """
    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)

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
            class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]        # randomly choose B one-hot for initialize
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1      # one-hot

        # Generate images.
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}     # unwrap kwargs, withdraw non-stated params
        have_ablation_kwargs = any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
        if 'randn_like' in sampler_kwargs:
            if sampler_kwargs['randn_like'] == 'db': sampler_kwargs['randn_like'] = rnd.randn_like
            else: sampler_kwargs['randn_like'] = rnd.multiGaussian_like
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
