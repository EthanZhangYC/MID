import torch
from torch.nn import Module
import torch.nn as nn
from .encoders.trajectron import Trajectron
from .encoders import dynamics as dynamic_module
import models.diffusion as diffusion
from models.diffusion import DiffusionTraj,VarianceSchedule
from models.traj_unet import Guide_UNet
import pdb

class AutoEncoder(Module):

    def __init__(self, config, encoder):
        super().__init__()
        self.config = config
        self.encoder = encoder
        
        if self.config.embed_latent and config.diffnet != "Guide_UNet":
            print('embedding latent')
            self.latent_embed = nn.Linear(config.encoder_dim,128)
            encoder_dim = 128
        else:
            self.latent_embed = None
            encoder_dim = config.encoder_dim
        

        if config.diffnet == "Guide_UNet":
            self.diffusion = DiffusionTraj(
                net = Guide_UNet(config),
                var_sched = VarianceSchedule(
                    num_steps=100,
                    beta_T=5e-2,
                    mode='linear'
                )
            )
        else:
            self.diffnet = getattr(diffusion, config.diffnet)
            self.diffusion = DiffusionTraj(
                net = self.diffnet(point_dim=2, context_dim=encoder_dim, tf_layer=config.tf_layer, residual=False, config=config),
                var_sched = VarianceSchedule(
                    num_steps=100,
                    beta_T=5e-2,
                    mode='linear'
                )
            )
        
    def encode(self, batch,node_type):
        z = self.encoder.get_latent(batch, node_type)
        return z
    
    def generate(self, latent, num_points, sample, bestof, flexibility=0.0, ret_traj=False, sampling="ddpm", step=100):
        #print(f"Using {sampling}")
        # dynamics = self.encoder.node_models_dict[node_type].dynamic
        # encoded_x = self.encoder.get_latent(batch, node_type)
        if self.latent_embed:
            latent = self.latent_embed(latent)
        predicted_y_vel = self.diffusion.sample(num_points, latent, sample, bestof, flexibility=flexibility, ret_traj=ret_traj, sampling=sampling, step=step)
        return predicted_y_vel.cpu().detach().numpy()
        # predicted_y_pos = dynamics.integrate_samples(predicted_y_vel)
        return predicted_y_pos.cpu().detach().numpy()
    
    def generate_ori(self, batch, node_type, num_points, sample, bestof,flexibility=0.0, ret_traj=False, sampling="ddpm", step=100):
        #print(f"Using {sampling}")
        dynamics = self.encoder.node_models_dict[node_type].dynamic
        encoded_x = self.encoder.get_latent(batch, node_type)
        predicted_y_vel =  self.diffusion.sample(num_points, encoded_x,sample,bestof, flexibility=flexibility, ret_traj=ret_traj, sampling=sampling, step=step)
        predicted_y_pos = dynamics.integrate_samples(predicted_y_vel)
        return predicted_y_pos.cpu().detach().numpy()

    def get_loss_ori(self, batch, node_type):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch

        feat_x_encoded = self.encode(batch,node_type) # B * 64
        loss = self.diffusion.get_loss(y_t.cuda(), feat_x_encoded)
        return loss
    
    def get_loss(self, batch, node_type=None, latent=None, img_feat=None, traj_feat=None):
        # (first_history_index,
        #  x_t, y_t, x_st_t, y_st_t,
        #  neighbors_data_st,
        #  neighbors_edge_value,
        #  robot_traj_st_t,
        #  map) = batch
        # feat_x_encoded = self.encode(batch, node_type) # B * 64
        # loss = self.diffusion.get_loss(y_t.cuda(), feat_x_encoded)
        if self.latent_embed:
            latent = self.latent_embed(latent)
        loss = self.diffusion.get_loss(batch.cuda(), latent, img_feat=img_feat, traj_feat=traj_feat)
        return loss
