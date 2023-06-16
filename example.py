import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
import torch.autograd.forward_ad as fwAD

#----------------------------------------------------------------------------
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


def generate_image_grid(
    network_pkl, dest_path,
    seed=0, gridw=8, gridh=8, device=torch.device('cuda'),
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=3,
    S_churn=0, S_min=0, S_max=50.0, S_noise=1,
):
    batch_size = gridw * gridh
    torch.manual_seed(seed)

    # Load network.
    print(f'Loading network from "{network_pkl}"...')
    with open(network_pkl, 'rb') as f:
        net = pickle.load(f).to(device)

    # Pick latents and labels.
    print(f'Generating {batch_size} images...')
    latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    class_labels = None
    if net.label_dim:
        class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    A = torch.tensor(sigma_max**(1/rho), dtype=torch.float64)
    B = torch.tensor(sigma_min**(1/rho) - sigma_max**(1/rho), dtype=torch.float64)

    # # Time step discretization, turn time-steps into sigma-schedule.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)  # [0,1,...,num_steps-1]
    sigmas = (A + step_indices/(num_steps-1)*B).pow(rho)
    sigmas = torch.cat([net.round_sigma(sigmas), torch.zeros_like(sigmas[:1])]) # t_steps[num_steps] = 0
    dt = torch.tensor(1/(num_steps-1), dtype=torch.float64, device=latents.device)

    x_next = latents.to(torch.float64) * sigmas[0]     # amplify to sigma_max variance
    for i, (sigma_cur, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])): # 0, ..., N-1
        x_cur = x_next
        dt = (sigma_next**(1/rho) - sigma_cur**(1/rho))/B      # dt>0

        # # increase nosie level except last iteration
        if i<num_steps-1:
            diffusion_coff = dt**((rho)/2)*((-B)**(rho)*sigma_cur).sqrt()
            noise = multiGaussian_like(x_cur, dt)
            x_cur = x_cur + diffusion_coff * noise[0]
            sigma_cur = (sigma_cur**2 + dt*diffusion_coff**2).sqrt()
            dt = (sigma_next**(1/rho) - sigma_cur**(1/rho))/B

        gamma=torch.tensor(0.0, dtype=torch.float64, device=latents.device)
        prod=torch.tensor(1.0, dtype=torch.float64, device=latents.device)
        for j in range(1, int(rho)+1):
            prod *= (B*(rho-j+1)/j)
            gamma += dt**(j-1)*prod*sigma_cur**((rho-j)/rho)

        # # 2nd order sampling.
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
            x_next = x_next + f_cur*gamma*dt

    # Save image grid.
    print(f'Saving image grid to "{dest_path}"...')
    image = (x_next * 127.5 + 128).clip(0, 255).to(torch.uint8)
    image = image.reshape(gridh, gridw, *image.shape[1:]).permute(0, 3, 1, 4, 2)
    image = image.reshape(gridh * net.img_resolution, gridw * net.img_resolution, net.img_channels)
    image = image.cpu().numpy()
    PIL.Image.fromarray(image, 'RGB').save(dest_path)
    print('Done.')

#----------------------------------------------------------------------------

def main():
    model_root = 'ckpts'
    generate_image_grid(f'{model_root}/edm-cifar10-32x32-cond-vp.pkl',   'cifar10-32x32.png',  num_steps=84)
    generate_image_grid(f'{model_root}/edm-ffhq-64x64-uncond-vp.pkl',    'ffhq-64x64.png',     num_steps=84)
    generate_image_grid(f'{model_root}/edm-imagenet-64x64-cond-adm.pkl', 'imagenet-64x64.png', num_steps=84)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
