MODEL_FOLDER="run/finetuned/cifar10/pruning_ratio_0.44/ddpm_cifar10_pruned_diff_pruning_thr_0.04_pruning_ratio_0.44_jm_1e-1"
OUTPUT_FOLDER="run/sample/cifar10/pruning_ratio_0.44/ddpm_cifar10_pruned_diff_pruning_thr_0.04_pruning_ratio_0.44_jm_1e-1_100k"

python -m torch.distributed.launch --nproc_per_node=2 --master_port 22230 --use_env ddpm_sample.py \
--batch_size 512 \
--total_samples 4096 \
--output_dir $OUTPUT_FOLDER \
--model_path $MODEL_FOLDER \
--pruned_model_ckpt "$MODEL_FOLDER/pruned/unet_ema_pruned-100000.pth" \
--ddim_steps 100 \
--skip_type uniform \
--seed 0
#
CUDA_VISIBLE_DEVICES=0  python fid_score.py  $OUTPUT_FOLDER  "$OUTPUT_FOLDER/fid_stats_cifar10.npz" --save-stats --device cuda:0 --batch-size 256
CUDA_VISIBLE_DEVICES=0  python fid_score.py  run/fid_stats_cifar10.npz "$OUTPUT_FOLDER/fid_stats_cifar10.npz"