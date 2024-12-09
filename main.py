from mid import MID
import argparse
import os
import yaml
# from pprint import pprint
from easydict import EasyDict
import numpy as np
import pdb
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of MID')
    parser.add_argument('--config', default='')
    parser.add_argument('--dataset', default='')
    parser.add_argument('--resume', default='')
    parser.add_argument('--traj_len', type=int, default=202)
    parser.add_argument('--job_dir', default='results/test')
    parser.add_argument('--embed_latent', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--proxy_train', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--proxy_eval', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--relative_xy', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--use_img', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--use_traj', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--not_filter_padding', action='store_true', help='whether to output attention in encoder')

    return parser.parse_args()


def main():
    torch.set_num_threads(4)
    # parse arguments and load config
    args = parse_args()
    with open(args.config) as f:
       config = yaml.safe_load(f)
    
    if not os.path.exists(args.job_dir):
        os.makedirs(args.job_dir)
        os.makedirs(args.job_dir+'/ckpt')
        os.makedirs(args.job_dir+'/ckpt_proxy')

    for k, v in vars(args).items():
       config[k] = v
    config["exp_name"] = args.config.split("/")[-1].split(".")[0]
    config["dataset"] = args.dataset[:-1]
    #pdb.set_trace()
    # if "cond1" in args.job_dir:
    #     config.encoder_dim = 1
    config = EasyDict(config)
    agent = MID(config)

    # keyattr = ["lr", "data_dir", "epochs", "dataset", "batch_size","diffnet"]
    # keys = {}
    # for k,v in config.items():
    #     if k in keyattr:
    #         keys[k] = v
    #
    # pprint(keys)

    sampling = "ddim"
    steps = 5

    if config["eval_mode"]:
        agent.eval(sampling, 100//step)
    elif args.proxy_train:
        print('proxy training')
        if args.relative_xy:
            print("relative_xy")
        agent.train_proxy()
    elif args.proxy_eval:
        print('proxy evaluate')
        assert len(args.dataset)>0
        if args.relative_xy:
            print("relative_xy")
        agent.eval_proxy(args.dataset, args.resume)
    else:
        agent.train()





if __name__ == '__main__':
    main()
