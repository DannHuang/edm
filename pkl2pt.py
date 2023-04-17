import torch
import pickle
import dnnlib
import argparse


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_dir', help='url to nvidia model zoo')
    parser.add_argument('--arch', default='ddpmpp', help='network architecture')
    parser.add_argument('--precond', default='edm', help='network preconditioning')
    parser.add_argument('--cbase')
    parser.add_argument('--cres')
    parser.add_argument('--augment', default=0.12)
    parser.add_argument('--dropout', default=0.13)
    parser.add_argument('--fp16', default=True)
    args = parser.parse_args()
    network_kwargs = dnnlib.EasyDict()
    assert args.pkl_dir is not None

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
    interface_kwargs = dict(img_resolution=32, img_channels=3, label_dim=10)
    
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
    net_dict = net.state_dict()
    net_list = list(net_dict.keys())
    print(len(net_list))
    print(net_list[:10])
    print(f'Loading network pkl file from "{args.pkl_dir}"...')
    with dnnlib.util.open_url(args.pkl_dir, verbose=True) as f:
        ema_net = pickle.load(f)['ema'].to('cpu')
        ema_dict = ema_net.state_dict()    # import torch?
        ema_list = list(ema_dict.keys())
        print(len(ema_list))
        print(ema_list[:10])
        # load state_dict to new NN and save model
    