import re
import torch
import pickle
import dnnlib
import argparse
import click
import torch.autograd.forward_ad as fwAD

'''
python recompileNN.py --pkl_dir=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl --arch=ddpmpp
python recompileNN.py --pkl_dir=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl --arch=adm
python recompileNN.py --pkl_dir=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl --arch=ddpmpp --cres=1,2,2,2 --dropout=0.05 --augment=0.15
'''

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

@click.command()

# Main options.
@click.option('--pkl_dir',        help='url to nvidia model zoo', metavar='DIR',                     type=str, required=True)
@click.option('--arch',           help='Network architecture', metavar='ddpmpp|ncsnpp|adm',          type=click.Choice(['ddpmpp', 'ncsnpp', 'adm']), default='ddpmpp', show_default=True)
@click.option('--precond',        help='Preconditioning & loss function', metavar='vp|ve|edm',       type=click.Choice(['vp', 've', 'edm']), default='edm', show_default=True)
@click.option('--cbase',          help='Channel multiplier  [default: varies]', metavar='INT',       type=int)
@click.option('--cres',           help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list)
@click.option('--augment',        help='Augment probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.12, show_default=True)
@click.option('--dropout',        help='Dropout probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.13, show_default=True)
@click.option('--fp16',           help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=True, show_default=True)

def main(**kwargs):
    args = dnnlib.EasyDict(kwargs)
    network_kwargs = dnnlib.EasyDict()
    assert args.pkl_dir is not None
    if args.pkl_dir == 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl':
        res=32
        num_channels=3
        label_dim=10
        model_dir = 'ckpts/edm-cifar10-32x32-cond-vp.pkl'
    if args.pkl_dir == 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl':
        res=64
        num_channels=3
        label_dim=1000
        model_dir = 'ckpts/edm-imagenet-64x64-cond-adm.pkl'
    if args.pkl_dir =='https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl':
        res=64
        num_channels=3
        label_dim=0
        model_dir = 'ckpts/edm-ffhq-64x64-uncond-vp.pkl'

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
    interface_kwargs = dict(img_resolution=res, img_channels=num_channels, label_dim=label_dim)
    # interface_kwargs = dict(img_resolution=32, img_channels=3, label_dim=10)

    print(f'Loading network pkl file from "{args.pkl_dir}"...')
    with dnnlib.util.open_url(args.pkl_dir, verbose=True) as f:
        ema_net = pickle.load(f)['ema'].to('cpu')
        # with fwAD.dual_level():
        #     dual_input = fwAD.make_dual(x_test, tan)
        #     dual_out = ema_net(dual_input, torch.tensor(0.5), torch.zeros(1,1000))

    # # load state_dict to new NN and save model
    ema_dict = ema_net.state_dict()
    ema_list = list(ema_dict.keys())
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
    net_dict = net.state_dict()
    net_list = list(net_dict.keys())
    assert len(ema_list)==len(net_list)
    net.load_state_dict(ema_dict)
    net = net.eval().requires_grad_(False)
    print(ema_list[:10])
    for name,param in net.named_parameters(): assert param.requires_grad==False
    print(net_list[:10])

    print(f'Wrting network to pkl file "{model_dir}"...')
    with open(model_dir, 'wb') as f:
        pickle.dump(net, f)

if __name__ == "__main__":
    main()

'''
test
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
print(torch.pow(res.to(torch.float32)-Jv.to(torch.float32), 2).sum())'''