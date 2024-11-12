
CUDA_VISIBLE_DEVICES=6 \
python main.py \
--config configs/baseline_our.yaml \
--job_dir results/1109_cond6_lr1e4_bs2048

CUDA_VISIBLE_DEVICES=8 \
python main.py \
--config configs/baseline_our.yaml \
--embed_latent \
--job_dir results/1109_cond1_lr1e4_bs256_embedlatent

CUDA_VISIBLE_DEVICES=4 \
python main.py \
--config configs/baseline_our.yaml \
--job_dir results/1111_proxy_lr1e3_n10_dim1024

CUDA_VISIBLE_DEVICES=9 \
python main.py \
--config configs/baseline_our.yaml \
--proxy_mode \
--relative_xy \
--job_dir results/1111_proxy_lr1e3_n10_dim1024_relxy




CUDA_VISIBLE_DEVICES=8 \
python main.py \
--config configs/baseline_our.yaml \
--traj_len 400 \
--job_dir results/1007_cond6_lr1e3_len400






--traj_len 200 \

CUDA_VISIBLE_DEVICES=7 \
python traj_generate.py \
--config configs/baseline_our.yaml \
--traj_len 200 \
--embed_latent \
--job_dir results/test