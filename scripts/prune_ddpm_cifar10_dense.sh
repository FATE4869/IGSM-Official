CUDA_VISIBLE_DEVICES=0 python ddpm_prune.py \
--dataset cifar10 \
--model_path google/ddpm-cifar10-32 \
--save_path run/pruned/cifar10/dense \
--pruning_ratio 0.0 \
--batch_size 128 \
--device cuda:0 \
--pruner diff-pruning \
--thr 0.04 \
--seed 0