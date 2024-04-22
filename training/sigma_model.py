import dnnlib
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            self.init_shift = torch.tensor(self.sigma_max**(1/self.rho), dtype = torch.float64)
            self.init_scale = torch.tensor(self.sigma_min**(1/self.rho) - self.sigma_max**(1/self.rho), dtype = torch.float64)
            step_indices = torch.arange(diffusion_length, dtype = torch.float64)    # [0,1,...,num_steps-1]
            sigmas = (self.init_shift + step_indices/(diffusion_length - 1) * self.init_scale).pow(self.rho)
            increments = (sigmas[1:]-sigmas[:-1]) / (sigma_min-sigma_max)           # dm_length - 1
            logits = increments.log()
            c = 1 - logits[-1]
            logits = logits + c
            self.init_vec = logits[:-1]             # less 1 degree of freedom due to soft-max. dm_length - 2
        else:
            self.init_vec=torch.randn(self.dm_length - 2, dtype = torch.float64)
        self.vec = torch.nn.Parameter(self.init_vec, requires_grad = True)

    def forward(self, positions):
        ph = torch.ones_like(self.init_vec[:1], dtype=torch.float64, device=self.vec.device)
        v = torch.cat([self.vec, ph])
        increment = F.softmax(v, dim = 0)
        percentage = torch.einsum('i,bji->bj', increment, positions)
        sigmas = percentage * (self.sigma_min - self.sigma_max) + self.sigma_max
        return sigmas

class finetune_wrapper(nn.Module):

    def __init__(self, diffusion_network_kwargs, interface_kwargs, sigma_model_kwargs, loss_mode='Dns'):
        super().__init__()
        self.diffusion_net = dnnlib.util.construct_class_by_name(**diffusion_network_kwargs, **interface_kwargs)
        self.sigma_model = dnnlib.util.construct_class_by_name(**sigma_model_kwargs)
        self.loss_mode = loss_mode

    def get_loss(self, images, labels, augment_pipe=None):
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