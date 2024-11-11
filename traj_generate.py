

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

def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of MID')
    parser.add_argument('--config', default='')
    parser.add_argument('--dataset', default='')
    parser.add_argument('--traj_len', type=int, default=200)
    parser.add_argument('--job_dir', default='results/test')

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
# config["exp_name"] = args.config.split("/")[-1].split(".")[0]
# config["dataset"] = args.dataset[:-1]
config = EasyDict(config)
agent = MID(config)
sampling = "ddim"
steps = 5
# agent.eval(sampling, 100//step)


train_data_loader = load_data(config.batch_size, config.traj_len)
for batch_data in train_data_loader:
    batch_data_x = batch_data[0]
    speed = batch_data_x[:,:,4]
    x0 = torch.ones_like(batch_data_x[:,:,:2])
    x0[:,:,1] = speed * torch.sin(batch_data_x[:,:,7]/180*np.pi)
    x0[:,:,0] = speed * torch.cos(batch_data_x[:,:,7]/180*np.pi)
    break


# n_steps = config.diffusion.num_diffusion_timesteps
# beta = torch.linspace(config.diffusion.beta_start,
#                           config.diffusion.beta_end, n_steps).cuda()
# alpha = 1. - beta
# alpha_bar = torch.cumprod(alpha, dim=0)
# lr = 2e-4  # Explore this - might want it lower when training on the full dataset

# eta=0.0
# timesteps=100
# skip = n_steps // timesteps
# seq = range(0, n_steps, skip)

# # # load head information for guide trajectory generation
# # batchsize = 500
# # head = np.load('heads.npy',
# #                    allow_pickle=True)
# # head = torch.from_numpy(head).float()
# # dataloader = DataLoader(head, batch_size=batchsize, shuffle=True, num_workers=4)


# # # # the mean and std of head information, using for rescaling
# # # # departure_time, trip_distance,  trip_time, trip_length, avg_dis, avg_speed
# # # hmean=[0, 10283.41600429,   961.66920921,   292.30299616,    36.02766493, 10.98568072]
# # # hstd=[1, 8782.599246414231, 379.41939897358264, 107.24874828393955, 28.749924691281066, 8.774629812281198]
# mean = np.array([104.07596303,   30.68085491])
# std = np.array([2.15106194e-02, 1.89193207e-02])
# # # # the original mean and std of trajectory length, using for rescaling the trajectory length
# len_mean = 292.30299616  # Chengdu
# len_std = 107.2487482839  # Chengdu




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
# head = np.array([[0.0000e+00],[1.0000e+00, ],[2.0000e+00, ],[3.0000e+00,]])
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
    new_traj = agent.model.generate(head, num_points=args.traj_len, sample=20, bestof=True, sampling=sampling, step=100//steps) # B * 20 * 12 * 2
    new_traj = new_traj[0,0] # (20, 4, 200, 2)
    # pdb.set_trace()
    
    # ims = []
    # n = x0.size(0)
    # x = x0
    # seq_next = [-1] + list(seq[:-1])
    # for i, j in zip(reversed(seq), reversed(seq_next)):
    #     t = (torch.ones(n) * i).to(x.device)
    #     next_t = (torch.ones(n) * j).to(x.device)
    #     with torch.no_grad():
    #         pred_noise = unet(x, t, head)
    #         x = p_xt(x, pred_noise, t, next_t, beta, eta)
    #         if i % 10 == 0:
    #             ims.append(x.cpu().squeeze(0))
    # trajs = ims[-1].cpu().numpy()
    # trajs = trajs[:,:2,:]
    # # resample the trajectory length
    # # for j in range(batchsize):
    # j=0
    # new_traj = resample_trajectory(trajs[j].T, lengths[j])
    # # new_traj = new_traj * std + mean
    
    # if input_speed:
    lat_min,lat_max = (0.0, 49.95356703911097)
    lon_min,lon_max = (0.0, 49.95356703911097)
    new_traj[:,0] = new_traj[:,0] * (lat_max-lat_min) + lat_min
    new_traj[:,1] = new_traj[:,1] * (lon_max-lon_min) + lon_min        
    # start_pt = [36,72]
    tmp_x = 36
    tmp_y = 72
    new_new_traj = []
    new_new_traj.append((tmp_x,tmp_y))
    for i in range(len(new_traj)):
        tmp_x += new_traj[i,0]*5/111/1000
        tmp_y += new_traj[i,1]*5/111/1000
        new_new_traj.append((tmp_x,tmp_y))
    new_traj = np.array(new_new_traj)
    # else:
    #     lat_min,lat_max = (18.249901, 55.975593)
    #     lon_min,lon_max = (-122.3315333, 126.998528)
    #     new_traj[:,0] = new_traj[:,0] * (lat_max-lat_min) + lat_min
    #     new_traj[:,1] = new_traj[:,1] * (lon_max-lon_min) + lon_min
    # # print(new_traj)
    Gen_traj.append(new_traj)


# try:
fig = plt.figure(figsize=(12,12))
for i in range(len(Gen_traj)):
    traj=Gen_traj[i]
    ax1 = fig.add_subplot(331+i)  
    ax1.plot(traj[:,0],traj[:,1],color='blue',alpha=0.1)
# except:
#     pdb.set_trace()
plt.tight_layout()
plt.savefig(filename)
plt.show()

# plt.figure(figsize=(8,8))
# for i in range(len(Gen_traj)):
#     traj=Gen_traj[i]
#     plt.plot(traj[:,0],traj[:,1],color='blue',alpha=0.1)
# plt.tight_layout()
# plt.savefig('Chengdu_traj.png')
# plt.show()