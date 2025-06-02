CUDA_VISIBLE_DEVICES=0 python ddpm_prune.py \
--dataset cifar10 \
--model_path google/ddpm-cifar10-32 \
--save_path run/pruned/cifar10/pruning_ratio_0.44/ddpm_cifar10_pruned_diff_pruning_thr_0.04 \
--pruning_ratio 0.3 \
--batch_size 128 \
--device cuda:0 \
--pruner diff-pruning \
--thr 0.04 \
--seed 0