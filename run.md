
CUDA_VISIBLE_DEVICES=9 \
python main.py \
--config configs/baseline_our.yaml \
--job_dir results/1006_cond6_lr5e4_bs256

CUDA_VISIBLE_DEVICES=6 \
python main.py \
--config configs/baseline_our.yaml \
--job_dir results/1006_cond6_lr1e3_bs1024