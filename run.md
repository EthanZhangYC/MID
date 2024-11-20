
CUDA_VISIBLE_DEVICES=6 \
python main.py \
--config configs/baseline_our.yaml \
--job_dir results/1109_cond6_lr1e4_bs2048

CUDA_VISIBLE_DEVICES=1 \
python main.py \
--config configs/baseline_our.yaml \
--embed_latent \
--use_img \
--job_dir results/1113_cond1_lr1e4_bs1024_embedlatent_useimg

CUDA_VISIBLE_DEVICES=6 \
python main.py \
--config configs/baseline_our.yaml \
--embed_latent \
--traj_len 650 \
--not_filter_padding \
--job_dir results/1120_cond8_lr1e4_bs1024_embedlatent_nofilterpad





--traj_len 200 \

CUDA_VISIBLE_DEVICES=6 \
python traj_generate.py \
--config configs/baseline_our.yaml \
--traj_len 650 \
--embed_latent \
--generate \
--job_dir results/test





CUDA_VISIBLE_DEVICES=7 \
python extract_tgt_feat.py 


















CUDA_VISIBLE_DEVICES=5 \
python main.py \
--config configs/baseline_our.yaml \
--proxy_mode \
--job_dir results/1111_proxy_lr1e3_n10_dim1024

CUDA_VISIBLE_DEVICES=5 \
python main.py \
--config configs/baseline_our.yaml \
--proxy_train \
--relative_xy \
--job_dir results/1111_proxy_lr1e3_n10_dim1024_relxy


CUDA_VISIBLE_DEVICES=5 \
python main.py \
--config configs/baseline_our.yaml \
--proxy_eval \
--resume /home/yichen/MID/results/1111_proxy_lr1e3_n10_dim1024/ckpt_proxy/unet_45000.pt \
--dataset /home/yichen/MID/1115_1109_cond6_lr1e4_bs256_embedlatent_epoch32000.npy \
--job_dir results/test