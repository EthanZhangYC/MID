
CUDA_VISIBLE_DEVICES=6 \
python main.py \
--config configs/baseline_our.yaml \
--job_dir results/1009_cond6_lr1e4_bs2048

CUDA_VISIBLE_DEVICES=8 \
python main.py \
--config configs/baseline_our.yaml \
--embed_latent \
--job_dir results/1009_cond1_lr1e4_bs256_embedlatent




CUDA_VISIBLE_DEVICES=8 \
python main.py \
--config configs/baseline_our.yaml \
--traj_len 400 \
--job_dir results/1007_cond6_lr1e3_len400







CUDA_VISIBLE_DEVICES=7 \
python traj_generate.py \
--config configs/baseline_our.yaml \
--traj_len 200 \
--job_dir results/test