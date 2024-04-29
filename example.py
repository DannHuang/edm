import os
import click
import pickle
import numpy as np
import torch
import PIL.Image
from generate import StackedRandomGenerator

#----------------------------------------------------------------------------

@click.group()
def main():
    """Create a grid of samples.

    Examples:

    \b
    # Generate 8x8 grid of samples
    python example.py --outdir=fid-tmp --seeds=0-49999 --subdirs \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    """

#----------------------------------------------------------------------------

@main.command()
@click.option('--model',         help='model',                  metavar='PKL',    type=str, required=True)
@click.option('--out',         help='out directory',                  metavar='PATH',    type=str, required=True)
@click.option('--gridw',         help='grid weight',            metavar='INT',    type=click.IntRange(min=1), default=8, show_default=True)
@click.option('--gridh',         help='grid height',  metavar='INT',    type=click.IntRange(min=1), default=8, show_default=True)

def finetune(model, out, gridw, gridh, device=torch.device('cuda')):
    batch_size = gridw * gridh

    # Load network.
    print(f'Loading network from "{model}"...')
    with open(model, 'rb') as f:
        data = pickle.load(f)
        net = data['ema'].to(device) if 'ema' in data else data.to(device)
        del data

    print(f'Generating {batch_size} images...')
    images = net.sample(batch_size=batch_size, class_idx=None, device=device)

    # # Save image grid.
    image = (images * 127.5 + 128).clip(0, 255).to(torch.uint8)
    image = image.reshape(gridh, gridw, *image.shape[1:]).permute(0, 3, 1, 4, 2)
    image = image.reshape(gridh * net.diffusion.img_resolution, gridw * net.diffusion.img_resolution, net.diffusion.img_channels)
    image = image.cpu().numpy()
    PIL.Image.fromarray(image, 'RGB').save(os.path.join(out, "finetune_sample.png"))
    print('Done.')

#----------------------------------------------------------------------------

@main.command()
@click.option('--model',         help='model',                  metavar='PKL',    type=str, required=True)
@click.option('--out',         help='out directory',                  metavar='PATH',    type=str, required=True)
@click.option('--gridw',         help='grid weight',            metavar='INT',    type=click.IntRange(min=1), default=8, show_default=True)
@click.option('--gridh',         help='grid height',  metavar='INT',    type=click.IntRange(min=1), default=8, show_default=True)
@click.option('--steps',         help='number of network forward passes',  metavar='INT',    type=click.IntRange(min=1), default=10, show_default=True)

def baseline(
    model, out, gridw, gridh, steps, device=torch.device('cuda'),
    sigma_min=0.002, sigma_max=80, rho=3,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    batch_size = gridw * gridh

    # Load network.
    print(f'Loading network from "{model}"...')
    with open(model, 'rb') as f:
        data = pickle.load(f)
        net = data['ema'].to(device) if 'ema' in data else data.to(device)
        del data

    # Pick latents and labels.
    print(f'Generating {batch_size} images...')
    seeds = [i for i in range(batch_size)]
    rnd = StackedRandomGenerator(device, seeds)
    latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    class_labels = None
    if net.label_dim:
        class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    A = torch.tensor(sigma_max**(1/rho), dtype=torch.float64)
    B = torch.tensor(sigma_min**(1/rho) - sigma_max**(1/rho), dtype=torch.float64)

    # # Time step discretization, turn time-steps into sigma-schedule.
    step_indices = torch.arange(steps, dtype=torch.float64, device=latents.device)  # [0,1,...,num_steps-1]
    sigmas = (A + step_indices/(steps-1)*B).pow(rho)
    sigmas = torch.cat([net.round_sigma(sigmas), torch.zeros_like(sigmas[:1])]) # t_steps[num_steps] = 0
    print(f'Sampling with sigmas: {[s.item() for s in sigmas.squeeze()]}')

    x_next = latents.to(torch.float64) * sigmas[0]     # amplify to sigma_max variance
    for i, (sigma_cur, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])): # 0, ..., N-1
        x_cur = x_next

        # # increase nosie level except last iteration
        # if i<num_steps-1:
        #     diffusion_coff = dt**((rho-1)/2)*((-B)**(rho)).sqrt()
        #     noise = multiGaussian_like(x_cur, dt)
        #     x_cur = x_cur + diffusion_coff * noise[0]
        #     sigma_cur = (sigma_cur**2 + dt*diffusion_coff**2).sqrt()
        #     dt = (sigma_next**(1/rho) - sigma_cur**(1/rho))/B

        # # Euler step.
        denoised = net(x_cur, sigma_cur, class_labels).to(torch.float64)
        eps = (x_cur - denoised) / sigma_cur  
        x_next = denoised + sigma_next * eps

    # # Save image grid.
    image = (x_next * 127.5 + 128).clip(0, 255).to(torch.uint8)
    image = image.reshape(gridh, gridw, *image.shape[1:]).permute(0, 3, 1, 4, 2)
    image = image.reshape(gridh * net.img_resolution, gridw * net.img_resolution, net.img_channels)
    image = image.cpu().numpy()
    PIL.Image.fromarray(image, 'RGB').save(os.path.join(out, "base_sample.png"))
    print('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
