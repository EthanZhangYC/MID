

import matplotlib.pyplot as plt
import argparse
import os
import yaml
# from pprint import pprint
from easydict import EasyDict
import numpy as np
import pdb
import torch
from models.autoencoder import AutoEncoder
from mid import load_data
import pickle
import scipy

def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of MID')
    parser.add_argument('--config', default='')
    parser.add_argument('--dataset', default='')
    parser.add_argument('--traj_len', type=int, default=200)
    parser.add_argument('--job_dir', default='results/test')
    parser.add_argument('--embed_latent', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--generate', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--use_img', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--use_traj', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--inter_seconds', type=int, default=50)

    return parser.parse_args()


class MID():
    def __init__(self, config):
        self.config = config
        torch.backends.cudnn.benchmark = True
        self._build_model()

    def eval(self, sampling, step):
        epoch = self.config.eval_at

        self.log.info(f"Sampling: {sampling} Stride: {step}")

        node_type = "PEDESTRIAN"
        eval_ade_batch_errors = []
        eval_fde_batch_errors = []
        ph = self.hyperparams['prediction_horizon']
        max_hl = self.hyperparams['maximum_history_length']

        for i, scene in enumerate(self.eval_scenes):
            print(f"----- Evaluating Scene {i + 1}/{len(self.eval_scenes)}")
            for t in tqdm(range(0, scene.timesteps, 10)):
                timesteps = np.arange(t,t+10)
                batch = get_timesteps_data(env=self.eval_env, scene=scene, t=timesteps, node_type=node_type, state=self.hyperparams['state'],
                               pred_state=self.hyperparams['pred_state'], edge_types=self.eval_env.get_edge_types(),
                               min_ht=7, max_ht=self.hyperparams['maximum_history_length'], min_ft=12,
                               max_ft=12, hyperparams=self.hyperparams)
                if batch is None:
                    continue
                test_batch = batch[0]
                nodes = batch[1]
                timesteps_o = batch[2]
                traj_pred = self.model.generate(test_batch, node_type, num_points=12, sample=20,bestof=True, sampling=sampling, step=step) # B * 20 * 12 * 2

                predictions = traj_pred
                predictions_dict = {}
                for i, ts in enumerate(timesteps_o):
                    if ts not in predictions_dict.keys():
                        predictions_dict[ts] = dict()
                    predictions_dict[ts][nodes[i]] = np.transpose(predictions[:, [i]], (1, 0, 2, 3))



                batch_error_dict = evaluation.compute_batch_statistics(predictions_dict,
                                                                       scene.dt,
                                                                       max_hl=max_hl,
                                                                       ph=ph,
                                                                       node_type_enum=self.eval_env.NodeType,
                                                                       kde=False,
                                                                       map=None,
                                                                       best_of=True,
                                                                       prune_ph_to_future=True)

                eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[node_type]['ade']))
                eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[node_type]['fde']))

        ade = np.mean(eval_ade_batch_errors)
        fde = np.mean(eval_fde_batch_errors)

        if self.config.dataset == "eth":
            ade = ade/0.6
            fde = fde/0.6
        elif self.config.dataset == "sdd":
            ade = ade * 50
            fde = fde * 50
        print(f"Sampling: {sampling} Stride: {step}")
        print(f"Epoch {epoch} Best Of 20: ADE: {ade} FDE: {fde}")
        #self.log.info(f"Best of 20: Epoch {epoch} ADE: {ade} FDE: {fde}")

    def _build_model(self):
        """ Define Model """
        config = self.config
        model = AutoEncoder(config, encoder=None)
        self.model = model.cuda()
        # if self.config.eval_mode:
        #     self.model.load_state_dict(self.checkpoint['ddpm'])
        # print("> Model built!")


torch.set_num_threads(4)
args = parse_args()
with open(args.config) as f:
    config = yaml.safe_load(f)
for k, v in vars(args).items():
    config[k] = v
config = EasyDict(config)
agent = MID(config)
sampling = "ddim"
steps = 5
# agent.eval(sampling, 100//step)


# train_data_loader = load_data(config)
# for batch_data in train_data_loader:
#     batch_data_x = batch_data[0]
#     speed = batch_data_x[:,:,4]
#     x0 = torch.ones_like(batch_data_x[:,:,:2])
#     x0[:,:,1] = speed * torch.sin(batch_data_x[:,:,7]/180*np.pi)
#     x0[:,:,0] = speed * torch.cos(batch_data_x[:,:,7]/180*np.pi)
#     break


# MTL
current_minmax = [
    # (45.230416, 45.9997262293), (-74.31479102, -72.81248199999999), 
    # (45.230416, 45.9997262293), (-74.31479102, -72.81248199999999),  \
    (0.9999933186918497, 1198.999998648651), # time
    (0.0, 50118.17550774085), # dist
    (0.0, 49.95356703911097), # speed
    (-9.99348698095659, 9.958323482935628), #acc
    (-39.64566646191948, 1433.3438889109589), #jerk
    (0.0, 359.95536847383516)] #bearing
# # geolife
# current_minmax = [(18.249901, 55.975593), (-122.3315333, 126.998528), (18.249901, 55.975593), (-122.3315333, 126.998528), \
#     (0.9999933186918497, 1198.999998648651), (0.0, 50118.17550774085), (0.0, 49.95356703911097), (-9.99348698095659, 9.958323482935628), (-39.64566646191948, 1433.3438889109589), (0.0, 359.95536847383516)]
                



# head = np.array([[0.0000e+00, 1.1301e-02, 3.2167e-01, 1.0000e+00, 5.6503e-05, 1.1917e-02],[0.0000e+00, 8.9842e-03, 1.3333e-01, 1.0000e+00, 4.4921e-05, 2.2534e-02],[0.0000e+00, 8.9842e-03, 1.3333e-01, 1.0000e+00, 4.4921e-05, 2.2534e-02],[0.0000e+00, 8.9842e-03, 1.3333e-01, 1.0000e+00, 4.4921e-05, 2.2534e-02]])
# head = np.array([[1.0000e+00, 2.8011e-02, 1.3333e-01, 1.0000e+00, 1.4006e-04, 7.0259e-02, 1.2100e+02, 1.2100e+02],[0.0000e+00, 8.9842e-03, 1.3333e-01, 1.0000e+00, 4.4921e-05, 2.2534e-02, 1.2100e+02, 1.2100e+02]])
# head = np.array([[2.0000e+00, 6.2662e-02, 1.4033e-01, 1.0000e+00, 3.1331e-04, 1.5628e-01, 1.2100e+02, 1.2100e+02],[0.0000e+00, 8.9842e-03, 1.3333e-01, 1.0000e+00, 4.4921e-05, 2.2534e-02, 1.2100e+02, 1.2100e+02]])
# head = np.array([[3.0000e+00, 3.2609e-02, 1.7300e-01, 1.0000e+00, 1.6305e-04, 7.8854e-02, 1.2100e+02, 1.2100e+02],[0.0000e+00, 8.9842e-03, 1.3333e-01, 1.0000e+00, 4.4921e-05, 2.2534e-02, 1.2100e+02, 1.2100e+02]])

head = np.array([
    [0.0000e+00, 1.1301e-02, 3.2167e-01, 1.0000e+00, 5.6503e-05, 1.1917e-02],
    [1.0000e+00, 2.8011e-02, 1.3333e-01, 1.0000e+00, 1.4006e-04, 7.0259e-02],
    [2.0000e+00, 6.2662e-02, 1.4033e-01, 1.0000e+00, 3.1331e-04, 1.5628e-01],
    [3.0000e+00, 3.2609e-02, 1.7300e-01, 1.0000e+00, 1.6305e-04, 7.8854e-02]
])


def old_list():
    model_dir_list=[
        "/home/yichen/MID/results/1006_cond6_lr1e3_bs256/ckpt/unet_100.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_bs256/ckpt/unet_200.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_bs256/ckpt/unet_400.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_bs256/ckpt/unet_800.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_bs256/ckpt/unet_1600.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_bs256/ckpt/unet_3200.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_bs256/ckpt/unet_6400.pt",
    ] 
    filename='1106mtl_mid_speed_lr1e3_bs256.png'

    model_dir_list=[
        "/home/yichen/MID/results/1006_cond6_lr1e3_bs1024/ckpt/unet_1000.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_bs1024/ckpt/unet_2000.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_bs1024/ckpt/unet_4000.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_bs1024/ckpt/unet_8000.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_bs2048/ckpt/unet_1000.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_bs2048/ckpt/unet_2000.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_bs2048/ckpt/unet_4000.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_bs2048/ckpt/unet_8000.pt",
    ] 
    filename='1106mtl_mid_speed_lr1e3_bs1024_bs2048.png'

    model_dir_list=[
        "/home/yichen/MID/results/1006_cond6_lr1e4_bs256/ckpt/unet_1000.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e4_bs256/ckpt/unet_2000.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e4_bs256/ckpt/unet_4000.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e4_bs256/ckpt/unet_8000.pt",
        "/home/yichen/MID/results/1006_cond6_lr5e4_bs256/ckpt/unet_1000.pt",
        "/home/yichen/MID/results/1006_cond6_lr5e4_bs256/ckpt/unet_2000.pt",
        "/home/yichen/MID/results/1006_cond6_lr5e4_bs256/ckpt/unet_4000.pt",
        "/home/yichen/MID/results/1006_cond6_lr5e4_bs256/ckpt/unet_8000.pt",
        "/home/yichen/MID/results/1006_cond6_lr5e4_bs256/ckpt/unet_11000.pt",
    ] 
    filename='1106mtl_mid_speed_lr1e4_lr5e4.png'

    model_dir_list=[
        "/home/yichen/MID/results/1007_cond1_lr1e3_bs256/ckpt/unet_1000.pt",
        "/home/yichen/MID/results/1007_cond1_lr1e3_bs256/ckpt/unet_2000.pt",
        "/home/yichen/MID/results/1007_cond1_lr1e3_bs256/ckpt/unet_4000.pt",
        "/home/yichen/MID/results/1007_cond1_lr1e3_bs256/ckpt/unet_8000.pt",
        "/home/yichen/MID/results/1007_cond1_lr1e3_bs256/ckpt/unet_16000.pt",
        "/home/yichen/MID/results/1007_cond1_lr1e3_bs256/ckpt/unet_32000.pt",
        "/home/yichen/MID/results/1007_cond1_lr1e3_bs256/ckpt/unet_40000.pt"
    ] 
    head = np.array([[0.0000e+00],[1.0000e+00, ],[2.0000e+00, ],[3.0000e+00,]])
    filename='1107mtl_mid_speed_cond1_class0.png'

    model_dir_list=[
        "/home/yichen/MID/results/1007_cond6_lr1e3_bs256/ckpt/unet_1000.pt",
        "/home/yichen/MID/results/1007_cond6_lr1e3_bs256/ckpt/unet_2000.pt",
        "/home/yichen/MID/results/1007_cond6_lr1e3_bs256/ckpt/unet_4000.pt",
        "/home/yichen/MID/results/1007_cond6_lr1e3_bs256/ckpt/unet_8000.pt",
        "/home/yichen/MID/results/1007_cond6_lr1e3_bs256/ckpt/unet_16000.pt",
        "/home/yichen/MID/results/1007_cond6_lr1e3_bs256/ckpt/unet_21000.pt",
    ] 
    filename='1107mtl_mid_speed_lr1e3_bs256_class0.png'

    model_dir_list=[
        "/home/yichen/MID/results/1006_cond6_lr1e3_len24/ckpt/unet_100.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_len24/ckpt/unet_200.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_len24/ckpt/unet_400.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_len24/ckpt/unet_1000.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_len24/ckpt/unet_2000.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_len24/ckpt/unet_4000.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_len24/ckpt/unet_8000.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_len24/ckpt/unet_16000.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_len24/ckpt/unet_30000.pt",
    ] 
    filename='1107mtl_mid_speed_len24.png'

    model_dir_list=[
        "/home/yichen/MID/results/1006_cond6_lr1e3_len50/ckpt/unet_100.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_len50/ckpt/unet_200.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_len50/ckpt/unet_400.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_len50/ckpt/unet_1000.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_len50/ckpt/unet_2000.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_len50/ckpt/unet_4000.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_len50/ckpt/unet_8000.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_len50/ckpt/unet_16000.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_len50/ckpt/unet_30000.pt",
    ] 
    filename='1107mtl_mid_speed_len50.png'

    model_dir_list=[
        "/home/yichen/MID/results/1006_cond6_lr1e3_len100/ckpt/unet_100.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_len100/ckpt/unet_200.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_len100/ckpt/unet_400.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_len100/ckpt/unet_1000.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_len100/ckpt/unet_2000.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_len100/ckpt/unet_4000.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_len100/ckpt/unet_8000.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_len100/ckpt/unet_16000.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_len100/ckpt/unet_30000.pt",
    ] 
    filename='1107mtl_mid_speed_len100.png'

    model_dir_list=[
        "/home/yichen/MID/results/1006_cond6_lr1e3_len400/ckpt/unet_1000.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_len400/ckpt/unet_2000.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_len400/ckpt/unet_4000.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_len400/ckpt/unet_8000.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_len400/ckpt/unet_16000.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_len400/ckpt/unet_32000.pt",
        "/home/yichen/MID/results/1006_cond6_lr1e3_len400/ckpt/unet_50000.pt",
    ] 
    filename='1107mtl_mid_speed_len400.png'

    model_dir_list=[
        "/home/yichen/MID/results/1008_cond6_lr1e3_bs256/ckpt/unet_1000.pt",
        "/home/yichen/MID/results/1008_cond6_lr1e3_bs256/ckpt/unet_2000.pt",
        "/home/yichen/MID/results/1008_cond6_lr1e3_bs256/ckpt/unet_4000.pt",
        "/home/yichen/MID/results/1008_cond6_lr1e3_bs256/ckpt/unet_8000.pt",
        "/home/yichen/MID/results/1008_cond6_lr1e3_bs256/ckpt/unet_16000.pt",
        "/home/yichen/MID/results/1008_cond6_lr1e3_bs256/ckpt/unet_32000.pt",
        "/home/yichen/MID/results/1008_cond6_lr1e3_bs256/ckpt/unet_50000.pt",
    ] 
    filename='1109mtl_mid_speed_lr1e3.png'

    model_dir_list=[
        "/home/yichen/MID/results/1008_cond6_lr1e4_bs256/ckpt/unet_1000.pt",
        "/home/yichen/MID/results/1008_cond6_lr1e4_bs256/ckpt/unet_2000.pt",
        "/home/yichen/MID/results/1008_cond6_lr1e4_bs256/ckpt/unet_4000.pt",
        "/home/yichen/MID/results/1008_cond6_lr1e4_bs256/ckpt/unet_8000.pt",
        "/home/yichen/MID/results/1008_cond6_lr1e4_bs256/ckpt/unet_16000.pt",
        "/home/yichen/MID/results/1008_cond6_lr1e4_bs256/ckpt/unet_32000.pt",
        "/home/yichen/MID/results/1008_cond6_lr1e4_bs256/ckpt/unet_50000.pt",
    ] 
    filename='1109mtl_mid_speed_lr1e4.png'

    model_dir_list=[
        "/home/yichen/MID/results/1008_cond6_lr5e4_bs256/ckpt/unet_1000.pt",
        "/home/yichen/MID/results/1008_cond6_lr5e4_bs256/ckpt/unet_2000.pt",
        "/home/yichen/MID/results/1008_cond6_lr5e4_bs256/ckpt/unet_4000.pt",
        "/home/yichen/MID/results/1008_cond6_lr5e4_bs256/ckpt/unet_8000.pt",
        "/home/yichen/MID/results/1008_cond6_lr5e4_bs256/ckpt/unet_16000.pt",
        "/home/yichen/MID/results/1008_cond6_lr5e4_bs256/ckpt/unet_32000.pt",
        "/home/yichen/MID/results/1008_cond6_lr5e4_bs256/ckpt/unet_50000.pt",
    ] 
    filename='1109mtl_mid_speed_lr5e4.png'


    model_dir_list=[
        "/home/yichen/MID/results/1009_cond1_lr1e4_bs256_embedlatent/ckpt/unet_1000.pt",
        "/home/yichen/MID/results/1009_cond1_lr1e4_bs256_embedlatent/ckpt/unet_2000.pt",
        "/home/yichen/MID/results/1009_cond1_lr1e4_bs256_embedlatent/ckpt/unet_4000.pt",
        "/home/yichen/MID/results/1009_cond1_lr1e4_bs256_embedlatent/ckpt/unet_8000.pt",
        "/home/yichen/MID/results/1009_cond1_lr1e4_bs256_embedlatent/ckpt/unet_16000.pt",
        "/home/yichen/MID/results/1009_cond1_lr1e4_bs256_embedlatent/ckpt/unet_32000.pt",
    ] 
    filename='1111mtl_mid_speed_embedlatent_cond1_class0.png'
    # filename='1111mtl_mid_speed_embedlatent_cond1_class0_ddpm.png'
    # head = np.array([[0.0000e+00],[1.0000e+00, ],[2.0000e+00, ],[3.0000e+00,]])


    model_dir_list=[
        "/home/yichen/MID/results/1009_cond6_lr1e4_bs256_embedlatent/ckpt/unet_1000.pt",
        "/home/yichen/MID/results/1009_cond6_lr1e4_bs256_embedlatent/ckpt/unet_2000.pt",
        "/home/yichen/MID/results/1009_cond6_lr1e4_bs256_embedlatent/ckpt/unet_4000.pt",
        "/home/yichen/MID/results/1009_cond6_lr1e4_bs256_embedlatent/ckpt/unet_8000.pt",
        "/home/yichen/MID/results/1009_cond6_lr1e4_bs256_embedlatent/ckpt/unet_16000.pt",
        "/home/yichen/MID/results/1009_cond6_lr1e4_bs256_embedlatent/ckpt/unet_32000.pt",
    ] 
    filename='1111mtl_mid_speed_embedlatent_cond6.png'



    model_dir_list=[
        "/home/yichen/MID/results/1109_cond1_lr1e4_bs256_embedlatent/ckpt/unet_4000.pt",
        "/home/yichen/MID/results/1109_cond1_lr1e4_bs256_embedlatent/ckpt/unet_8000.pt",
        "/home/yichen/MID/results/1109_cond1_lr1e4_bs256_embedlatent/ckpt/unet_16000.pt",
        "/home/yichen/MID/results/1109_cond1_lr1e4_bs256_embedlatent/ckpt/unet_32000.pt",
    ] 
    n_data = 10000
    head = torch.from_numpy(np.random.randint(0,4,[n_data,1])).float()

    model_dir_list=[
        "/home/yichen/MID/results/1109_cond6_lr1e4_bs256_embedlatent/ckpt/unet_4000.pt",
        "/home/yichen/MID/results/1109_cond6_lr1e4_bs256_embedlatent/ckpt/unet_8000.pt",
        "/home/yichen/MID/results/1109_cond6_lr1e4_bs256_embedlatent/ckpt/unet_16000.pt",
        "/home/yichen/MID/results/1109_cond6_lr1e4_bs256_embedlatent/ckpt/unet_32000.pt",
    ] 


# 1124
model_dir_list=[
    "/home/yichen/MID/results/1120_cond6_lr1e4_bs1024_embedlatent_len202/ckpt/unet_1000.pt",
    "/home/yichen/MID/results/1120_cond6_lr1e4_bs1024_embedlatent_len202/ckpt/unet_2000.pt",
    "/home/yichen/MID/results/1120_cond6_lr1e4_bs1024_embedlatent_len202/ckpt/unet_4000.pt",
    "/home/yichen/MID/results/1120_cond6_lr1e4_bs1024_embedlatent_len202/ckpt/unet_8000.pt",
    "/home/yichen/MID/results/1120_cond6_lr1e4_bs1024_embedlatent_len202/ckpt/unet_16000.pt",
] 


# model_dir_list=[
#     "/home/yichen/MID/results/1120_cond8_lr1e4_bs1024_embedlatent_len202/ckpt/unet_1000.pt",
#     "/home/yichen/MID/results/1120_cond8_lr1e4_bs1024_embedlatent_len202/ckpt/unet_2000.pt",
#     "/home/yichen/MID/results/1120_cond8_lr1e4_bs1024_embedlatent_len202/ckpt/unet_4000.pt",
#     "/home/yichen/MID/results/1120_cond8_lr1e4_bs1024_embedlatent_len202/ckpt/unet_8000.pt",
#     "/home/yichen/MID/results/1120_cond8_lr1e4_bs1024_embedlatent_len202/ckpt/unet_16000.pt",
# ] 

# 1203
model_dir_list=[
    "/home/yichen/MID/results/1125_cond6_lr1e4_bs1024_embedlatent_len600/ckpt/unet_1000.pt",
    "/home/yichen/MID/results/1125_cond6_lr1e4_bs1024_embedlatent_len600/ckpt/unet_2000.pt",
    "/home/yichen/MID/results/1125_cond6_lr1e4_bs1024_embedlatent_len600/ckpt/unet_4000.pt",
    "/home/yichen/MID/results/1125_cond6_lr1e4_bs1024_embedlatent_len600/ckpt/unet_8000.pt",
    "/home/yichen/MID/results/1125_cond6_lr1e4_bs1024_embedlatent_len600/ckpt/unet_16000.pt",
] 
filename='1204mtl_mid_speed_embedlatent_cond6.png'
head = np.array([
    [0.0000e+00, 6.1595e+01, 2.0000e-01, 600., 1.0266e-01, 1.0266e-01],
    [1.0000e+00, 6.3451e+00, 2.0000e-01, 600., 1.0575e-02, 1.0575e-02],
    [2.0000e+00, 4.4909e+01, 2.0000e-01, 600., 7.4848e-02, 7.4848e-02],
    [3.0000e+00, 1.7232e+01, 1.3867e-01, 550., 4.1423e-02, 4.1423e-02],
])

model_dir_list=[
    "/home/yichen/MID/results/1204_cond6_lr5e3_bs1024_embedlatent_len102_correctbr/ckpt/unet_1000.pt",
    "/home/yichen/MID/results/1204_cond6_lr5e3_bs1024_embedlatent_len102_correctbr/ckpt/unet_2000.pt",
    "/home/yichen/MID/results/1204_cond6_lr5e3_bs1024_embedlatent_len102_correctbr/ckpt/unet_4000.pt",
    "/home/yichen/MID/results/1204_cond6_lr5e3_bs1024_embedlatent_len102_correctbr/ckpt/unet_8000.pt",
    "/home/yichen/MID/results/1204_cond6_lr5e3_bs1024_embedlatent_len102_correctbr/ckpt/unet_16000.pt",
] 
filename='1205mtl_mid_speed_embedlatent_cond6_len100.png'


model_dir_list=[
    "/home/yichen/MID/results/1205_cond6_lr1e3_bs1024_embedlatent_len102_correctbr_guideunet/ckpt/unet_1000.pt",
    "/home/yichen/MID/results/1205_cond6_lr1e3_bs1024_embedlatent_len102_correctbr_guideunet/ckpt/unet_2000.pt",
    "/home/yichen/MID/results/1205_cond6_lr1e3_bs1024_embedlatent_len102_correctbr_guideunet/ckpt/unet_4000.pt",
    "/home/yichen/MID/results/1205_cond6_lr1e3_bs1024_embedlatent_len102_correctbr_guideunet/ckpt/unet_8000.pt",
    "/home/yichen/MID/results/1205_cond6_lr1e3_bs1024_embedlatent_len102_correctbr_guideunet/ckpt/unet_16000.pt",
    "/home/yichen/MID/results/1205_cond6_lr1e3_bs1024_embedlatent_len102_correctbr_guideunet/ckpt/unet_24000.pt",
] 
filename='1206mtl_mid_speed_embedlatent_cond6_len100_guideunet.png'
head = np.array([
    [0.0000e+00, 6.1595e+01, 2.0000e-01, 102., 1.0266e-01, 1.0266e-01],
    [1.0000e+00, 6.3451e+00, 2.0000e-01, 102., 1.0575e-02, 1.0575e-02],
    [2.0000e+00, 4.4909e+01, 2.0000e-01, 102., 7.4848e-02, 7.4848e-02],
    [3.0000e+00, 1.7232e+01, 1.3867e-01, 95., 4.1423e-02, 4.1423e-02],
])


if not args.generate:
    print(filename)
    # x0 = torch.randn(batchsize, 2, config.data.traj_length).cuda()
    head = torch.from_numpy(head).float().cuda()

    Gen_traj = []
    Gen_head = []
    for model_dir in model_dir_list:
        ckpt_dir = model_dir
        print(ckpt_dir)
        checkpoint = torch.load(ckpt_dir)
        agent.model.load_state_dict(checkpoint)
        new_traj = agent.model.generate(head, num_points=args.traj_len, sample=2, bestof=True)#, sampling=sampling, step=100//steps) # B * 20 * 12 * 2
        new_traj = new_traj[0,0] # (20, 4, 200, 2)
        
        if args.inter_seconds==5:
            lat_min,lat_max = (0.0, 49.95356703911097)
            lon_min,lon_max = (0.0, 49.95356703911097)
        elif args.inter_seconds==1:
            lat_min,lat_max = (0.0, 80.0)
            lon_min,lon_max = (0.0, 80.0)
        new_traj[:,0] = new_traj[:,0] * (lat_max-lat_min) + lat_min
        new_traj[:,1] = new_traj[:,1] * (lon_max-lon_min) + lon_min   
             
        # start_pt = [36,72]
        tmp_x = 36
        tmp_y = 72
        new_new_traj = []
        new_new_traj.append((tmp_x,tmp_y))
        for i in range(len(new_traj)):
            tmp_x += new_traj[i,0]*args.inter_seconds/111/1000
            tmp_y += new_traj[i,1]*args.inter_seconds/111/1000
            new_new_traj.append((tmp_x,tmp_y))
        new_traj = np.array(new_new_traj)
        Gen_traj.append(new_traj)

    fig = plt.figure(figsize=(12,12))
    for i in range(len(Gen_traj)):
        traj=Gen_traj[i]
        ax1 = fig.add_subplot(331+i)  
        ax1.plot(traj[:,0],traj[:,1],color='blue',alpha=0.1)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

else:
    
    # head = np.load("/home/yichen/MID/1114_heads_tgt.npy")
    head = np.load("/home/yichen/MID/1203_heads_tgt.npy")
    head = torch.from_numpy(head).float()
    print("Head shape: ", head.shape) # [55866, 6]

    bs = 128
    for model_dir in model_dir_list:
        filename = 'outputs/1204_%s_epoch%s_len%d_%ds_1203heads'%(
            model_dir.split('/')[-3],
            model_dir.split('_')[-1],
            args.traj_len,
            args.inter_seconds
        )
        filename = filename.replace(".pt","")
        
        ckpt_dir = model_dir
        print(ckpt_dir)
        checkpoint = torch.load(ckpt_dir)
        agent.model.load_state_dict(checkpoint)
        val_loader = torch.utils.data.DataLoader(head, batch_size=bs, drop_last=False)
        
        Gen_traj = []
        Gen_feat = []
        Gen_head = head
        eps = 1e-9
        for idx,batch_head in enumerate(val_loader):
            # if idx>5:
            #     break
            
            batch_head = batch_head.cuda()
            new_v = agent.model.generate(batch_head, num_points=args.traj_len, sample=2, bestof=True, sampling=sampling, step=100//steps) # (2, bs, 200, 2)
            new_v = new_v[0] # (bs, 200, 2)

            if args.inter_seconds == 5:
                v_min,v_max = (0.0, 49.95356703911097)
            elif args.inter_seconds == 1:
                v_min,v_max = (0.0, 80.0)
            else:
                raise NotImplemented
            
            new_v = new_v * (v_max-v_min) + v_min
            new_st = new_v*args.inter_seconds/111/1000 
            
            n_timestep = new_v.shape[1]-2
            abs_v = np.sqrt(np.square(new_v).sum(axis=-1))
            new_dist = abs_v * args.inter_seconds
            new_a = np.diff(abs_v, axis=1)/args.inter_seconds                
            new_br = (np.arctan(new_v[:,:,1]/new_v[:,:,0]+eps)/np.pi*180 + 360) % 360
            
            abs_v = abs_v[:,:n_timestep]
            new_dist = new_dist[:,:n_timestep]
            new_a = new_a[:,:n_timestep]
            
            
            def compute_bearing(p1, p2):
                y = math.sin(math.radians(p2[1]) - math.radians(p1[1])) * math.radians(math.cos(p2[0]))
                x = math.radians(math.cos(p1[0])) * math.radians(math.sin(p2[0])) - \
                    math.radians(math.sin(p1[0])) * math.radians(math.cos(p2[0])) \
                    * math.radians(math.cos(p2[1]) - math.radians(p1[1]))
                # Convert radian from -pi to pi to [0, 360] degree
                return (math.atan2(y, x) * 180. / math.pi + 360) % 360

            def compute_bearing_rate(bearing1, bearing2):
                return abs(bearing1 - bearing2)
            
            
            if args.inter_seconds == 5:
                new_br = new_br[:,:n_timestep]
                new_j = np.diff(new_a, axis=1)/args.inter_seconds
                is_real = np.ones_like(abs_v)
                feat_array = np.stack([np.ones_like(abs_v)*args.inter_seconds,new_dist,abs_v,new_a,new_j,new_br,is_real], axis=-1)
            elif args.inter_seconds == 1:
                new_br_rate = np.diff(new_br, axis=1)/args.inter_seconds
                new_br = new_br[:,:n_timestep]
                new_br_rate = new_br_rate[:,:n_timestep]
                feat_array = np.stack([np.ones_like(abs_v)*args.inter_seconds,new_dist,abs_v,new_a,new_br,new_br_rate], axis=-1)
            Gen_feat.append(feat_array)
            
            tmp_x = 36+np.random.randn()/100
            tmp_y = 72+np.random.randn()/100
            cum_st = np.cumsum(new_st, axis=1)
            new_traj = np.ones([new_v.shape[0], args.traj_len, 2]) * [tmp_x, tmp_y]
            new_traj = new_traj + cum_st

            new_traj = new_traj[:,:n_timestep]
            Gen_traj.append(new_traj)

            
        Gen_traj = np.concatenate(Gen_traj,axis=0)
        Gen_feat = np.concatenate(Gen_feat,axis=0)
        
        # normalize 
        if args.inter_seconds == 5:
            for index, item in enumerate(current_minmax):
                pdb.set_trace()
                if index == 0 or index == 5: # time and bearing
                    continue
                Gen_feat[:, :, index] = (Gen_feat[:, :, index] - item[0])/(item[1] - item[0]) 
        elif args.inter_seconds == 1:
            
            
            # def running_mean_convolve(x, N):
            #     return np.convolve(x, np.ones(N) / float(N), 'valid')
            # def running_mean_cumsum(x, N):
            #     cumsum = np.cumsum(np.insert(x, 0, 0))
            #     return (cumsum[N:] - cumsum[:-N]) / float(N)
            # def running_mean_uniform_filter1d(x, N):
            #     return scipy.ndimage.uniform_filter1d(x, N, mode='constant', origin=-(N//2))[:-(N-1)]

            N=10
            pad_array = np.tile(Gen_feat[:,-1:,:],[1,N+1,1])
            Gen_feat = np.concatenate([Gen_feat,pad_array], axis=1)
            cumsum = np.cumsum(Gen_feat, axis=1)
            cumsum = np.concatenate([np.zeros([cumsum.shape[0],1,6]),cumsum], axis=1)
            Gen_feat = (cumsum[:,N:] - cumsum[:,:-N]) / float(N)
            print(Gen_feat.shape)
            
            max_list = np.array([5. , 80. , 80. , 79.62211779 , 179.98863883, 179.96395057])
            min_list = np.array([0. , 0.  , 0.  , -72.77655734, 0.          , 0.          ])
            for i in range(6):
                Gen_feat[:,:,i] = (Gen_feat[:,:,i]-min_list[i])/(max_list[i]-min_list[i])
        else:
            raise NotImplemented
                
        with open(filename+".npy", "wb") as f:
            pickle.dump([Gen_traj, Gen_feat, Gen_head], f)

            
        fig,ax = plt.subplots()
        for i in range(len(Gen_traj)):
            traj = Gen_traj[i]
            plt.plot(traj[:,0],traj[:,1],color='blue',alpha=0.1)
        plt.tight_layout()
        plt.savefig(filename+".png")
        plt.show()
        