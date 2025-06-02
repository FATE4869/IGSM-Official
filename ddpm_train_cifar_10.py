"""
IGSM finetuning code of "Improved Geometric and Sensitivity Matching for Finetuning Pruned Diffusion Models"
Copyright (c) 2024-2025 University of Washington. Developed in UW NeuroAI Lab by Yang Zheng.
"""

import argparse
import logging
import math
import os, sys
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
import torchvision
from tqdm.auto import tqdm
import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel, DDIMPipeline, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import is_accelerate_version, is_tensorboard_available, is_wandb_available
import utils
from scipy.linalg import qr
from scipy.integrate import solve_ivp
import numpy as np
import pdb

logger = get_logger(__name__, log_level="INFO")


def compute_Jv(model, x, t, v):
    bs = x.shape[0]
    x.requires_grad = True
    v = v.to(x.device)
    output = model(x, t).sample.view(bs, -1)
    output_sum = torch.sum(output * v)
    Jv = torch.autograd.grad(outputs=output_sum, inputs=x,
                             retain_graph=False, create_graph=False, only_inputs=True)[0]
    return Jv.view(bs, -1)


# def compute_vTJv(model, x, t, v_i, v_j):
#     bs = x.shape[0]
#     x.requires_grad_(True)
#     Jv_j = compute_Jv(model, x, t, v_j)
#     vTJv = torch.sum(v_i.view(bs, -1) * Jv_j.view(bs, -1), dim=1)
#     return vTJv

def otd_evolution(V, model, x, t, k):
    bs = x.shape[0]
    JVs = [compute_Jv(model, x, t, V[:, :, i]) for i in range(k)]

    JV = torch.stack(JVs, dim=-1)  # JV = [bs, d, k]
    # JV = torch.stack([compute_Jv(model, x, t, V[:, :, i]) for i in range(k)], dim=-1)  # Compute J*V
    # B = torch.zeros((bs, k, k)).to(x.device)  # Initialize B = V^T J V
    # for i in range(k):
    #     for j in range(k):
    #         B[:, i, j] = torch.sum(V[:, :, i] * JVs[j], dim=1)
    # Compute B: [bs, k, k] = V^T J V
    B = torch.einsum('bdk,bdl->bkl', V, JV)  # fast!

    dV = JV - torch.matmul(V, B)  # OTD evolution: dV/dt = J V - V B

    # for i in range(k):
    #     for j in range(k):
    #         B[:, i, j] = compute_vTJv(model, x, t, V[:, :, i], V[:, :, j])  # Compute B_ij = v_i^T J v_j
    B = B.detach()
    JV = JV.detach()
    # dV = JV - torch.matmul(V, B)  # OTD evolution: dV/dt = J V - V B

    return dV


def orthonormalize(V0):
    bs, d, k = V0.shape
    if k == 1:
        # If k == 1, we can just normalize the vector
        return torch.nn.functional.normalize(V0)

    # Orthonormalize the modes using QR decomposition for each batch
    V0_orthonormal = torch.zeros_like(V0)

    for i in range(bs):
        Q, _ = torch.linalg.qr(V0[i])  # Apply QR on the [c*h*w, k] slice for each batch
        V0_orthonormal[i] = Q
    V0 = V0_orthonormal
    return V0


def evolve_otd_modes(model, x, t, k, T, num_steps=1, dt=0.01, prob=0.5):
    """
    Evolves the OTD modes over multiple time steps.
    Args:
        model: The diffusion model.
        x: Input to the model.
        k: Number of OTD modes.
        T: Total time for evolution.
        num_steps: Number of time steps.
        dt: Time step size.
        prob: Probability of not updating the OTD modes.
    Returns:
        t: Array of time points.
        V: Evolved OTD modes at each time point.
    """
    # Initialize random orthonormal OTD modes
    bs, c, h, w = x.shape
    V0 = orthonormalize(torch.randn(bs, c * h * w, k, device=x.device))
    # with prob not updating v
    if np.random.uniform() < prob:
        return V0
    for i in range(num_steps):
        t = torch.clamp(t, 0, T)
        dv = otd_evolution(V0, model, x, t, k)
        V0 = V0 + dv * dt
        V0 = orthonormalize(V0)
        t -= 1
    return V0


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pruned_model_ckpt", type=str, default=None)
    parser.add_argument("--project_name", type=str, default="ddpm")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that HF Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="The config of the UNet model to train, leave as None to use standard DDPM configuration.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset` is specified."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ddpm-model-64",
        help="The output directory where the checkpoints will be written.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default='./cache',
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="The number of images to generate for evaluation."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )
    parser.add_argument(
        "--checkpoint_id",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--load_ema",
        action="store_true",
        default=False,
    )
    parser.add_argument("--num_iters", type=int, default=10000)
    parser.add_argument(
        "--save_model_steps", type=int, default=1000, help="How often to save the model during training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.0, help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument("--kd_alpha", type=float, default=0.0, help="weight for knowledge distillation loss.")
    parser.add_argument("--ssm_alpha", type=float, default=0.0, help="weight for sliced score matching loss.")
    parser.add_argument("--jm_alpha", type=float, default=0.0, help="weight for jacobian matching loss.")
    parser.add_argument("--otd_steps", type=int, default=0, help="number of steps for OTD update.")
    parser.add_argument("--otd_prob", type=float, default=0.5, help="probability of not updating OTD.")
    parser.add_argument("--noisy_images_epsilon", type=float, default=1e-4, help="Epsilon value for noisy images.")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
    parser.add_argument("--ema_max_decay", type=float, default=0.999, help="The maximum decay magnitude for EMA.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_private_repo", action="store_true", help="Whether or not to create a private repository."
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help=(
            "Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai)"
            " for experiment tracking and logging of model metrics and model checkpoints"
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddim_num_inference_steps", type=int, default=100)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # if args.dataset is None and args.train_data_dir is None:
    #     raise ValueError("You must specify either a dataset name from the hub or a train data directory.")

    return args


def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    if args.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")
    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb
        # Name of the run, if not set will be set to the timestamp
        import datetime
        if args.project_name is None:
            args.project_name = f"{args.output_dir.split('/')[-1]}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation and wandb initialization.
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.logger == "wandb":
            wandb.init(
                project='Diff-Pruning',  # Name of the project
                entity='university_of_washington_4869',  # Entity/organization name (if any)
                name=args.project_name,  # Name of the run, if not set will be set to the timestamp
                config=args
            )

    # Loading dense model
    model_dense = torch.load('run/pruned/cifar10/dense/pruned/unet_pruned.pth', map_location='cpu').eval()
    # model_dense = torch.load(
    #     '/mmfs1/gscratch/shlneuroai/zheng94/prunerseeker/run/pruned/cifar10/dense/pruned/unet_pruned.pth',
    #     map_location='cpu').eval()

    # Loading pruned model
    if os.path.isdir(args.model_path):
        if args.pruned_model_ckpt is not None:
            print("Loading pruned model from {}".format(args.pruned_model_ckpt))
            unet = torch.load(args.pruned_model_ckpt, map_location='cpu').eval()
        else:
            print("Loading model from {}".format(args.model_path))
            subfolder = 'unet' if os.path.isdir(os.path.join(args.model_path, 'unet')) else None
            unet = UNet2DModel.from_pretrained(args.model_path, subfolder=subfolder).eval()
        pipeline = DDPMPipeline(
            unet=unet,
            scheduler=DDPMScheduler.from_pretrained(args.model_path, subfolder="scheduler")
        )
    # Loading standard model
    else:
        print("Loading pretrained model from {}".format(args.model_path))
        pipeline = DDPMPipeline.from_pretrained(
            args.model_path,
        )
    model = pipeline.unet

    noise_scheduler = pipeline.scheduler

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    dataset = utils.get_dataset(args.dataset)
    logger.info(f"Dataset size: {len(dataset)}")
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )
    num_epochs = math.ceil(args.num_iters / len(train_dataloader) * args.gradient_accumulation_steps)

    # Create EMA for the model.
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=False,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNet2DModel,
            model_config=model.config,
        )

    # Initialize the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_model.to(accelerator.device)
        model_dense.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_epochs *= accelerator.num_processes
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Total optimization steps = {args.num_iters}")
    logger.info(f"  Noisy image epsilon = {args.noisy_images_epsilon}")

    global_step = 0
    first_epoch = 0
    # save the shell command
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, 'run.sh'), 'w') as f:
            f.write('python ' + ' '.join(sys.argv))

    # setup dropout
    if args.dropout > 0:
        utils.set_dropout(model, args.dropout)

    if accelerator.is_main_process and args.logger == "wandb":
        wandb.init(
            project='Diff-Pruning',  # Name of the project
            entity='university_of_washington_4869',  # Entity/organization name (if any)
            name=args.project_name,  # Name of the run, if not set will be set to the timestamp
            config=args
        )
    # generate images before training
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(model).eval()
        if args.use_ema:
            ema_model.store(unet.parameters())
            ema_model.copy_to(unet.parameters())
        pipeline = DDIMPipeline(
            unet=unet,
            scheduler=DDIMScheduler(num_train_timesteps=args.ddpm_num_steps)
        )
        pipeline.scheduler.set_timesteps(args.ddim_num_inference_steps)
        images = pipeline(
            batch_size=args.eval_batch_size,
            num_inference_steps=args.ddim_num_inference_steps,
            output_type="numpy",
        ).images
        if args.use_ema:
            ema_model.restore(unet.parameters())
        os.makedirs(os.path.join(args.output_dir, 'vis'), exist_ok=True)
        torchvision.utils.save_image(torch.from_numpy(images).permute([0, 3, 1, 2]),
                                     os.path.join(args.output_dir, 'vis', 'before_training.png'))
        images_processed = (images * 255).round().astype("uint8")
        if args.logger == "tensorboard":
            if is_accelerate_version(">=", "0.17.0.dev0"):
                tracker = accelerator.get_tracker("tensorboard", unwrap=True)
            else:
                tracker = accelerator.get_tracker("tensorboard")
            # tracker.add_images("After Pruning", images_processed.transpose(0, 3, 1, 2), 0)
        elif args.logger == "wandb":
            # Upcoming `log_images` helper coming in https://github.com/huggingface/accelerate/pull/962/files
            accelerator.get_tracker("wandb").log(
                {"After Pruning": [wandb.Image(img) for img in images_processed], "epoch": 0},
                step=global_step,
            )
        del unet
        del pipeline
    accelerator.wait_for_everyone()
    # Train!
    os.makedirs(os.path.join(args.output_dir, 'vis'), exist_ok=True)
    print(f"dt for OTD: {args.dt}")
    for epoch in range(first_epoch, num_epochs):
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        mse_loss_at_epochs = 0.0
        ssm_loss_at_epochs = 0.0
        kd_loss_at_epochs = 0.0
        jac_loss_at_epochs = 0.0

        loss_at_epochs = 0.0
        for step, batch in enumerate(train_dataloader):
            model.train()
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch:  # and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            if isinstance(batch, (list, tuple)):
                clean_images = batch[0]
            else:
                clean_images = batch
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bsz = clean_images.shape[0]

            # The standard training procedure in diffusers
            # timesteps = torch.randint(
            #    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            # ).long()

            # Our experiements were conduct on https://github.com/ermongroup/ddim/blob/main/runners/diffusion.py
            timesteps = torch.randint(
                low=0, high=noise_scheduler.config.num_train_timesteps, size=(bsz // 2 + 1,)
            ).to(clean_images.device)
            timesteps = torch.cat([timesteps, noise_scheduler.config.num_train_timesteps - timesteps - 1], dim=0)[:bsz]

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                optimizer.zero_grad()

                ssm_loss, ssm_loss_ = 0.0, 0.0
                kd_loss, kd_loss_ = 0.0, 0.0
                jac_loss, jac_loss_ = 0.0, 0.0
                batch_size, c, h, w = noisy_images.shape

                if args.kd_alpha > 0:
                    noisy_images.requires_grad = True
                    vectors = evolve_otd_modes(model_dense, noisy_images, t=timesteps, k=1, T=1000,
                                               num_steps=args.otd_steps, dt=args.dt, prob=args.otd_prob)
                    # Compute KD loss

                    y = model(noisy_images, timesteps).sample.view(batch_size, -1)
                    mse_loss = (noise.view(batch_size, -1) - y).square().sum(dim=(-1)).mean(dim=0)

                    yd = model_dense(noisy_images, timesteps).sample.view(batch_size, -1)
                    kd_loss = (yd - y).square().sum(dim=-1).mean(dim=0)
                    kd_loss_ = kd_loss.detach().item()

                    if args.jm_alpha > 0:
                        y_v = torch.sum(y.unsqueeze(-1) * vectors)
                        yd_v = torch.sum(yd.unsqueeze(-1) * vectors)

                        J_v = torch.autograd.grad(y_v, noisy_images, create_graph=True)[0].view(batch_size, -1, 1)
                        Jd_v = torch.autograd.grad(yd_v, noisy_images, create_graph=True)[0].view(batch_size, -1,
                                                                                                  1).detach()
                        # pdb.set_trace()
                        jac_loss = (Jd_v.square().sum((1, 2)) - J_v.square().sum((1, 2))).square().mean()
                        jac_loss = args.jm_alpha * jac_loss
                        jac_loss_ = jac_loss.detach().item()

                    if args.ssm_alpha > 0:
                        ssm_loss = torch.sum((vectors * J_v).view(batch_size, -1), dim=-1).mean(dim=0)
                        ssm_loss_ = ssm_loss.detach().item()
                else:
                    model_output = model(noisy_images, timesteps).sample
                    mse_loss = (noise - model_output).square().sum(dim=(1, 2, 3)).mean(dim=0)

                mse_loss_ = mse_loss.detach().item()

                mse_loss_at_epochs += mse_loss_
                ssm_loss_at_epochs += ssm_loss_
                kd_loss_at_epochs += kd_loss_
                jac_loss_at_epochs += jac_loss_
                loss = mse_loss + ssm_loss + kd_loss + jac_loss

                loss_at_epochs += loss.item()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

            logs = {"mse_loss": mse_loss_, "ssm_loss": ssm_loss_,
                    "kd_loss": kd_loss_, "jac_loss": jac_loss_,
                    "loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            if accelerator.is_main_process and args.logger == "wandb":
                wandb.log(logs, step=global_step)
            accelerator.log(logs, step=global_step)

            # Save the model & generate sample images
            if global_step % args.save_model_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    # save the model
                    unet = accelerator.unwrap_model(model).eval()
                    unet.zero_grad()
                    os.makedirs(os.path.join(args.output_dir, 'pruned'), exist_ok=True)
                    torch.save(unet, os.path.join(args.output_dir, 'pruned', 'unet_pruned.pth'.format(global_step)))
                    torch.save(unet, os.path.join(args.output_dir, 'pruned', 'unet_pruned-{}.pth'.format(global_step)))
                    if args.use_ema:
                        ema_model.store(unet.parameters())
                        ema_model.copy_to(unet.parameters())
                        torch.save(unet,
                                   os.path.join(args.output_dir, 'pruned', 'unet_ema_pruned.pth'.format(global_step)))
                        torch.save(unet, os.path.join(args.output_dir, 'pruned',
                                                      'unet_ema_pruned-{}.pth'.format(global_step)))
                    pipeline = DDPMPipeline(
                        unet=unet,
                        scheduler=noise_scheduler,
                    )
                    pipeline.save_pretrained(args.output_dir)

                    # generate images
                    logger.info("Sampling images...")
                    pipeline = DDIMPipeline(
                        unet=unet,
                        scheduler=DDIMScheduler(num_train_timesteps=args.ddpm_num_steps)
                    )
                    pipeline.scheduler.set_timesteps(args.ddim_num_inference_steps)
                    images = pipeline(
                        batch_size=args.eval_batch_size,
                        num_inference_steps=args.ddim_num_inference_steps,
                        output_type="numpy",
                    ).images

                    if args.use_ema:
                        ema_model.restore(unet.parameters())
                    torchvision.utils.save_image(torch.from_numpy(images).permute([0, 3, 1, 2]),
                                                 os.path.join(args.output_dir, 'vis',
                                                              'iter-{}.png'.format(global_step)))
                    # denormalize the images and save to tensorboard
                    images_processed = (images * 255).round().astype("uint8")

                    if args.logger == "tensorboard":
                        if is_accelerate_version(">=", "0.17.0.dev0"):
                            tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                        else:
                            tracker = accelerator.get_tracker("tensorboard")
                        tracker.add_images("test_samples", images_processed.transpose(0, 3, 1, 2), global_step)
                    elif args.logger == "wandb":
                        # Upcoming `log_images` helper coming in https://github.com/huggingface/accelerate/pull/962/files
                        accelerator.get_tracker("wandb").log(
                            {"test_samples": [wandb.Image(img) for img in images_processed], "steps": global_step},
                            step=global_step,
                        )
                    del unet
                    del pipeline

            if global_step > args.num_iters:
                progress_bar.close()
                accelerator.wait_for_everyone()
                accelerator.end_training()
                return

        progress_bar.close()
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print("\n\n" + "#" * 50)
            print("Epoch: {}/{}".format(epoch + 1, num_epochs))
            print(f"mse_loss_sum: {mse_loss_at_epochs / num_update_steps_per_epoch:.4f}")
            print(f"ssm_loss_sum: {ssm_loss_at_epochs / num_update_steps_per_epoch:.4f}")
            print(f"kd_loss_sum: {kd_loss_at_epochs / num_update_steps_per_epoch:.4f}")
            print(f"jac_loss_sum: {jac_loss_at_epochs / num_update_steps_per_epoch:.4f}")
            print(f"loss_sum: {loss_at_epochs / num_update_steps_per_epoch:.4f}")

            print("#" * 50 + "\n\n")
        logs = {"mse_loss_sum": mse_loss_at_epochs / num_update_steps_per_epoch,
                "ssm_loss_at_epochs": ssm_loss_at_epochs / num_update_steps_per_epoch,
                "kd_loss_at_epochs": kd_loss_at_epochs / num_update_steps_per_epoch,
                "jac_loss_at_epochs": jac_loss_at_epochs / num_update_steps_per_epoch,
                "loss_at_epochs": loss_at_epochs / num_update_steps_per_epoch,
                "lr": lr_scheduler.get_last_lr()[0],
                "epoch": epoch + 1}
        if accelerator.is_main_process and args.logger == "wandb":
            wandb.log(logs, step=global_step)
        accelerator.log(logs, step=global_step)
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
