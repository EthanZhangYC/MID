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
    parser.add_argument('--traj_len', type=int, default=200)
    parser.add_argument('--job_dir', default='results/test')
    parser.add_argument('--embed_latent', action='store_true', help='whether to output attention in encoder')
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

    for k, v in vars(args).items():
       config[k] = v
    config["exp_name"] = args.config.split("/")[-1].split(".")[0]
    config["dataset"] = args.dataset[:-1]
    #pdb.set_trace()
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
    else:
        agent.train()





if __name__ == '__main__':
    main()
