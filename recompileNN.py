import torch
import pickle
import dnnlib
import argparse
import torch.autograd.forward_ad as fwAD

'''
python recompileNN.py --pkl_dir=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl --arch=adm --dropout=0.10 --augment=0 --fp16=1
python recompileNN.py --pkl_dir=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl --arch=adm --dropout=0.10 --augment=0 --fp16=1
'''

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_dir', help='url to nvidia model zoo')
    parser.add_argument('--arch', default='ddpmpp', help='network architecture')
    parser.add_argument('--precond', default='edm', help='network preconditioning')
    parser.add_argument('--cbase')
    parser.add_argument('--cres')
    parser.add_argument('--augment', default=0.12, type=float)
    parser.add_argument('--dropout', default=0.13, type=float)
    parser.add_argument('--fp16', default=True)
    args = parser.parse_args()
    network_kwargs = dnnlib.EasyDict()
    assert args.pkl_dir is not None
    # if args.pkl_dir == 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl':
    #     args

    if args.arch == 'ddpmpp':
        network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[2,2,2])
    elif args.arch == 'ncsnpp':
        network_kwargs.update(model_type='SongUNet', embedding_type='fourier', encoder_type='residual', decoder_type='standard')
        network_kwargs.update(channel_mult_noise=2, resample_filter=[1,3,3,1], model_channels=128, channel_mult=[2,2,2])
    else:       
        assert args.arch == 'adm'
        network_kwargs.update(model_type='DhariwalUNet', model_channels=192, channel_mult=[1,2,3,4])
    
    if args.precond == 'vp':
        network_kwargs.class_name = 'training.networks.VPPrecond'
    elif args.precond == 've':
        network_kwargs.class_name = 'training.networks.VEPrecond'
    else:
        assert args.precond == 'edm'
        network_kwargs.class_name = 'training.networks.EDMPrecond'
    if args.cbase is not None:
        network_kwargs.model_channels = args.cbase
    if args.cres is not None:
        network_kwargs.channel_mult = args.cres
    if args.augment:
        augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', p=args.augment)
        network_kwargs.augment_dim = 9
    network_kwargs.update(dropout=args.dropout, use_fp16=args.fp16)
    interface_kwargs = dict(img_resolution=64, img_channels=3, label_dim=1000)
    # interface_kwargs = dict(img_resolution=32, img_channels=3, label_dim=10)
    model_dir = 'ckpts/edm-cifar10-32x32-cond-vp.pkl'

    print(f'Loading network pkl file from "{args.pkl_dir}"...')
    with dnnlib.util.open_url(args.pkl_dir, verbose=True) as f:
        ema_net = pickle.load(f)['ema'].to('cpu')
        # with fwAD.dual_level():
        #     dual_input = fwAD.make_dual(x_test, tan)
        #     dual_out = ema_net(dual_input, torch.tensor(0.5), torch.zeros(1,1000))

    # # load state_dict to new NN and save model
    ema_dict = ema_net.state_dict()
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
    net.load_state_dict(ema_dict)
    net = net.eval().requires_grad_(False)
    # ema_list = list(ema_dict.keys())
    # print(len(ema_list))
    # print(ema_list[:10])
    for name,param in net.named_parameters(): assert param.requires_grad==False
    net_dict = net.state_dict()
    net_list = list(net_dict.keys())
    # print(len(net_list))
    # print(net_list[:10])
    print(f'Wrting network to pkl file "{model_dir}"...')
    with open(model_dir, 'wb') as f:
        pickle.dump(net, f)
    exit()

    with open(model_dir, 'rb') as f:
            net = pickle.load(f)
    x_test = torch.randn(1,3,64,64, dtype=torch.float32)
    tan = torch.randn_like(x_test, dtype=x_test.dtype)
    t_cur=torch.tensor(0.5, dtype=x_test.dtype)
    class_labels=torch.zeros(1,1000, dtype=x_test.dtype)
    with fwAD.dual_level():
        dual_input = fwAD.make_dual(x_test, tan)
        dual_out = net(dual_input, t_cur, class_labels)
        y, Jv = fwAD.unpack_dual(dual_out)
    
    # y = net(x_test, t_cur, class_labels).to(x_test.dtype)
    # y, Jv = torch.func.jvp(net, (x_test, torch.tensor(0.5), torch.zeros(1,1000)), (tan, torch.tensor(0.0), torch.zeros(1,1000, dtype=torch.float32)))
    
    x_test.requires_grad_()
    output = ema_net(x_test, t_cur, class_labels).to(x_test.dtype)
    # print(((output-y)**2).sum().sqrt())
    res = torch.zeros_like(x_test, dtype=x_test.dtype)
    for i in range(3):
        for j in range(64):
            for k in range(64):
                output[0][i][j][k].backward(retain_graph=True)
                # print(x_test.grad)
                # input()
                # print(((x_test.grad)*tan).sum())
                # print(Jv[0][i][j][k])
                res[0][i][j][k]=((x_test.grad)*tan).sum()
                x_test.grad.zero_()
    print(torch.allclose(res.to(torch.float32), Jv.to(torch.float32)))
    print(torch.pow(res.to(torch.float32)-Jv.to(torch.float32), 2).sum())