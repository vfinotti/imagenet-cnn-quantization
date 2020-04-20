import argparse
from utils import misc, quant, selector
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from collections import OrderedDict

def main():
    parser = argparse.ArgumentParser(description='PyTorch SVHN Example')
    parser.add_argument('--type', default='squeezenet_v1', help='|'.join(selector.known_models))
    parser.add_argument('--batch-size', type=int, default=1, help='input batch size for training')
    parser.add_argument('--ngpu', type=int, default=8, help='number of gpus to use')
    parser.add_argument('--use-gpu', dest='use_gpu', action='store_true')
    parser.add_argument('--use-cpu', dest='use_gpu', action='store_false')
    parser.set_defaults(use_gpu=True)
    parser.add_argument('--seed', type=int, default=117, help='random seed')
    parser.add_argument('--model-root', default='~/.torch/models/', help='folder to save the model')
    parser.add_argument('--data-root', default='./datasets', help='folder to save the model')
    parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
    parser.add_argument('--input-size', type=int, default=224, help='input size of image')
    parser.add_argument('--n-sample', type=int, default=20, help='number of samples to infer the scaling factor')
    parser.add_argument('--param-bits', type=int, default=30, help='bit-width for parameters')
    parser.add_argument('--bn-bits', type=int, default=30, help='bit-width for running mean and std')
    parser.add_argument('--fwd-bits', type=int, default=30, help='bit-width for layer output')
    parser.add_argument('--overflow-rate', type=float, default=0.0, help='overflow rate')
    args = parser.parse_args()

    assert args.use_gpu in [False, True]
    args.use_gpu = torch.cuda.is_available() if args.use_gpu else False
    args.ngpu = torch.cuda.device_count()
    misc.ensure_dir(args.logdir)
    args.model_root = misc.expand_user(args.model_root)
    args.data_root = misc.expand_user(args.data_root)
    args.input_size = 299 if 'inception' in args.type else args.input_size
    print("=================FLAGS==================")
    for k, v in args.__dict__.items():
        print('{}: {}'.format(k, v))
    print("========================================")

    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)

    # load model and dataset fetcher
    model_raw, ds_fetcher, is_imagenet = selector.select(args.type, model_root=args.model_root, cuda=args.use_gpu)
    args.ngpu = args.ngpu if is_imagenet else 1

    # quantize parameters
    if args.param_bits < 32:
        state_dict = model_raw.state_dict()
        state_dict_quant = OrderedDict()
        sf_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'running' in k:
                if args.bn_bits >=32:
                    print("Ignoring {}".format(k))
                    state_dict_quant[k] = v
                    continue
                else:
                    bits = args.bn_bits
            else:
                bits = args.param_bits

            temp_sf = quant.get_scalling_factor(v, overflow_rate=args.overflow_rate)
            v_quant  = quant.linear_quantize(v, temp_sf, bits=args.param_bits, return_type='float')
            state_dict_quant[k] = v_quant
            sf_dict[k] = temp_sf
            print(k, bits)
        model_raw.load_state_dict(state_dict_quant)

    # quantize forward activation
    if args.fwd_bits < 32:
        model_raw = quant.duplicate_model_with_quant(model_raw, bits=args.fwd_bits, overflow_rate=args.overflow_rate,
                                                     counter=args.n_sample)
        print(model_raw)
        val_ds_tmp = ds_fetcher(1, data_root=args.data_root, train=False, input_size=args.input_size)
        misc.eval_model(model_raw, val_ds_tmp, ngpu=1, n_sample=args.n_sample, is_imagenet=is_imagenet, cuda=args.use_gpu)

    # eval model
    val_ds = ds_fetcher(args.batch_size, data_root=args.data_root, train=False, input_size=args.input_size)
    acc1, acc5 = misc.eval_model(model_raw, val_ds, ngpu=args.ngpu, is_imagenet=is_imagenet, cuda=args.use_gpu)

    # print sf
    print(model_raw)
    res_str = "type={}, param_bits={}, bn_bits={}, fwd_bits={}, overflow_rate={}, acc1={:.4f}, acc5={:.4f}".format(
        args.type, args.param_bits, args.bn_bits, args.fwd_bits, args.overflow_rate, acc1, acc5)
    print(res_str)
    with open('acc1_acc5.txt', 'a') as f:
        f.write(res_str + '\n')


if __name__ == '__main__':
    main()
