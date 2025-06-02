import os
import logging
import pdb
import time
import glob
import math
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.utils.data as data
from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import noise_estimation_loss, noise_estimation_kd_loss, noise_estimation_kd_jac_loss
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
import torchvision.utils as tvu
import wandb
<<<<<<< HEAD
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
=======
>>>>>>> 31c62ac905f5796de46a8b0a761dd76feea6b65a

def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):

    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config

        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = self.betas.shape[0]

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = self.alphas.cumprod(dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), self.alphas_cumprod[:-1]], dim=0
        )
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = self.betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = self.posterior_variance.clamp(min=1e-20).log()
        # placeholder for model and teacher
        self.model = None
        self.teacher = None
        self.build_model()

    def build_model(self):
        args, config = self.args, self.config
        model = Model(config)
        if args.load_pruned_model is not None:
            print("Loading pruned model from {}".format(args.load_pruned_model))
            states = torch.load(args.load_pruned_model, map_location='cpu')

            if isinstance(states, torch.nn.Module): # a simple pruned model 
                model = torch.load(args.load_pruned_model, map_location='cpu')
            elif isinstance(states, list): # pruned model and training states
                model = states[0]
                if args.use_ema and self.config.model.ema:
                    print("Loading EMA")
                    ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                    ema_helper.register(model)
                    ema_helper.load_state_dict(states[-1])
                    ema_helper.ema(model)
                    self.ema_helper = ema_helper
                else:
                    self.ema_helper = None
            else:
                raise NotImplementedError
            self.model = model
        elif args.restore_from is not None and os.path.isfile(args.restore_from):
<<<<<<< HEAD
            pdb.set_trace()
=======
            # pdb.set_trace()
>>>>>>> 31c62ac905f5796de46a8b0a761dd76feea6b65a
            ckpt = args.restore_from
            print("Loading checkpoint {}".format(ckpt))
            states = torch.load(
                ckpt,
                map_location='cpu',
            )
            if isinstance(states[0], torch.nn.Module):
                model = states[0]
                model = model.to(self.device)
            else:
                model = model.to(self.device)
                model.load_state_dict(states[0], strict=True) 

            if args.use_ema and self.config.model.ema:
                print("Loading EMA")
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
                self.ema_helper = ema_helper
            else:
                self.ema_helper = None
        
        elif not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                ckpt = os.path.join(self.args.log_path, "ckpt.pth")
                states = torch.load(
                    ckpt,
                    map_location=self.config.device,
                )
            else:
                ckpt = os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    )
                states = torch.load(
                    ckpt,
                    map_location=self.config.device,
                )
            print("Loading checkpoint {}".format(ckpt))

            if isinstance(states[0], torch.nn.Module):
                model = states[0]
                model = model.to(self.device)
            else:
                model = model.to(self.device)
                model.load_state_dict(states[0], strict=True)

            if args.use_ema and self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
                self.ema_helper = ema_helper
            else:
                self.ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            elif self.config.data.dataset == "CELEBA":
                name = 'celeba'
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            states = torch.load(ckpt, map_location=self.device)
            if isinstance(states, (list,tuple)):
                model.load_state_dict(states[0])
            else:
                model.load_state_dict(states)
            model.to(self.device)
        self.model = model
        # self.model = CheckpointWrapper(model)

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        if self.args.kd:
            self.teacher = torch.load(self.args.teacher_model, map_location='cpu')
            self.teacher.to(device)

        args, config = self.args, self.config
        dataset, _ = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        # Prepare model and optimizer
        model = self.model
        optimizer = get_optimizer(self.config, model.parameters())
        if args.use_ema and self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        # Resume training
        start_epoch, step, resume_step = 0, 0, 0
        num_update_steps_per_epoch = math.floor(len(train_loader) / (accelerator.num_processes * config.training.gradient_accumulation_steps))

        if args.restore_from is not None and os.path.isfile(args.restore_from):
            ckpt = args.restore_from
            print("Resuming training from checkpoint: {}".format(ckpt))
            states = torch.load(ckpt, map_location='cpu')
            if isinstance(states[0], torch.nn.Module):
                model = states[0]
                optimizer = get_optimizer(self.config, model.parameters())  # rebuild optimizer
            else:
                model.load_state_dict(states[0])
            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            resume_step = states[4]
            # resume_step = step % num_update_steps_per_epoch
            if self.config.model.ema:
                ema_helper.load_state_dict(states[-1])

        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
        self.model = model

        if ema_helper is not None:
            ema_helper.to(accelerator.device)

        # if accelerator.is_main_process:
        #     unwrapped_model = accelerator.unwrap_model(model)
        #     unwrapped_model.eval()
        #     if ema_helper is not None:
        #         ema_helper.store(unwrapped_model.parameters())
        #         ema_helper.copy_to(unwrapped_model.parameters())
            # Log initial visualization before training
            # with torch.no_grad():
            #     n = config.sampling.batch_size
            #     x = torch.randn(
            #         n,
            #         config.data.channels,
            #         config.data.image_size,
            #         config.data.image_size,
            #         device=accelerator.device,
            #     )
            #     x = self.sample_image(x, unwrapped_model)
            #     x = inverse_data_transform(config, x)
            #     grid = tvu.make_grid(x)
            #     tvu.save_image(grid, os.path.join(args.log_path, 'vis', 'Init.png'))
            #     # tb_logger.add_image('Before Training', grid, global_step=0)
            #
            #     # Log the image to wandb
            #     wandb.log({"Before Training": wandb.Image(grid)})

            # if args.use_ema:
            #     ema_helper.restore(unwrapped_model.parameters())
        # print(len(train_loader))
        # num_epochs = math.ceil(config.training.n_iters / len(train_loader))
        total_batch_size = config.training.batch_size * accelerator.num_processes * config.training.gradient_accumulation_steps

        num_epochs = math.ceil(config.training.n_iters / num_update_steps_per_epoch)
        if accelerator.is_main_process:
            print("***** Running training *****")
            print(f"  Num examples = {len(dataset)}")
            print(f"  Instantaneous batch size per device = {config.training.batch_size}")
            print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
            print(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
            print(f"  Num Epochs = {num_epochs}")
            print(f"  Total optimization steps = {config.training.n_iters}")
<<<<<<< HEAD
=======
            print(f"  Current step = {step}")
>>>>>>> 31c62ac905f5796de46a8b0a761dd76feea6b65a

        for epoch in range(start_epoch, num_epochs):
            progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")
            data_start = time.time()
            data_time = 0
            loss_at_epoch = 0.0
            mse_at_epoch = 0.0
            kd_at_epoch = 0.0
            jac_at_epoch = 0.0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                # Skip already-completed steps when resuming training
                # if args.restore_from and epoch == start_epoch and i <= resume_step:
                #     progress_bar.update(1)
                #     continue

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                a = self.alphas_cumprod.index_select(0, t).view(-1, 1, 1, 1)

                x = x * a.sqrt() + e * (1.0 - a).sqrt()
<<<<<<< HEAD

                if self.args.jm:
                    loss, mse_loss, kd_loss, jac_loss = noise_estimation_kd_jac_loss(model, self.teacher, x, t, e,
                                                                                     noise_weight=1.0,
                                                                                     kd_weight=self.args.kd_weight,
                                                                                     jm_weight=self.args.jm_weight)
                    loss_ = loss.detach().item()
                    mse_loss_ = mse_loss.detach().item()
                    kd_loss_ = kd_loss.detach().item()
                    jac_loss_ = jac_loss.detach().item()
                elif self.args.kd:
                    loss, mse_loss, kd_loss = noise_estimation_kd_loss(model, self.teacher, x, t, e,
                                                                       noise_weight=1.0,
                                                                       kd_weight=self.args.kd_weight)
                    loss_ = loss.detach().item()
                    mse_loss_ = mse_loss.detach().item()
                    kd_loss_ = kd_loss.detach().item()
                    jac_loss_ = 0.0
                else:
                    loss = noise_estimation_loss(model, x, t, e)
                    mse_loss_ = loss.detach().item()
                    kd_loss_ = 0.0
                    jac_loss_ = 0.0
                loss_at_epoch += loss_
                mse_at_epoch += mse_loss_
                kd_at_epoch += kd_loss_
                jac_at_epoch += jac_loss_

                if step % self.config.training.validation_freq == 0:
                    logging.info(
                        f"step: {step} (Ep={epoch}/{self.config.training.n_epochs}, Iter={i}/{len(train_loader)}), "
                        f"loss: {loss_}, mse_loss: {mse_loss_}, kd_loss: {kd_loss_}, jac_loss: {jac_loss_}"
                    )
=======
                # import pdb
                # pdb.set_trace()
                with accelerator.accumulate(model):
                    
                    if self.args.jm:
                        loss, mse_loss, kd_loss, jac_loss = noise_estimation_kd_jac_loss(model, self.teacher, x, t, e,
                                                                                        noise_weight=1.0,
                                                                                        kd_weight=self.args.kd_weight,
                                                                                        jm_weight=self.args.jm_weight)
                        loss_ = loss.detach().item()
                        mse_loss_ = mse_loss.detach().item()
                        kd_loss_ = kd_loss.detach().item()
                        jac_loss_ = jac_loss.detach().item()
                    elif self.args.kd:
                        loss, mse_loss, kd_loss = noise_estimation_kd_loss(model, self.teacher, x, t, e,
                                                                        noise_weight=1.0,
                                                                        kd_weight=self.args.kd_weight)
                        loss_ = loss.detach().item()
                        mse_loss_ = mse_loss.detach().item()
                        kd_loss_ = kd_loss.detach().item()
                        jac_loss_ = 0.0
                    else:
                        loss = noise_estimation_loss(model, x, t, e)
                        loss_ = loss.detach().item()
                        mse_loss_ = loss_
                        kd_loss_ = 0.0
                        jac_loss_ = 0.0
                    loss_at_epoch += loss_
                    mse_at_epoch += mse_loss_
                    kd_at_epoch += kd_loss_
                    jac_at_epoch += jac_loss_
                    accelerator.backward(loss)
>>>>>>> 31c62ac905f5796de46a8b0a761dd76feea6b65a

                    if accelerator.sync_gradients:
<<<<<<< HEAD
                        accelerator.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
                except Exception:
                    pass
                optimizer.step()

                loss = loss.detach()
                mse_loss = mse_loss.detach()
                kd_loss = kd_loss.detach()
                jac_loss = jac_loss.detach()
                torch.cuda.empty_cache()

                import gc
                gc.collect()

                if args.use_ema and self.config.model.ema:
                    ema_helper.update(model)

                progress_bar.update(1)
                step += 1

                accelerator.wait_for_everyone()

                logs = {"loss": loss_, "mse_loss": mse_loss_, "kd_loss": kd_loss_, "jac_loss": jac_loss_, "step": step}

                if accelerator.is_main_process and args.logger == "wandb":
                    wandb.log(logs, step=step)
=======
                        # accelerator.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
                        optimizer.step()
                        optimizer.zero_grad()
                if accelerator.sync_gradients:
                    if args.use_ema and self.config.model.ema:
                        ema_helper.update(model)

                    progress_bar.update(1)
                    step += 1
                    # accelerator.wait_for_everyone()
                    if accelerator.is_local_main_process and (step % self.config.training.validation_freq == 0):
                            logging.info(
                                f"step: {step} (Ep={epoch}/{self.config.training.n_epochs}, Iter={i}/{num_update_steps_per_epoch}), "
                                f"loss: {loss_:.2f}, mse_loss: {mse_loss_:.2f}, kd_loss: {kd_loss_:.2f}, jac_loss: {jac_loss_:.2f}"
                            )
                    logs = {"loss": loss_, "mse_loss": mse_loss_, "kd_loss": kd_loss_, "jac_loss": jac_loss_, "step": step}

                    if accelerator.is_main_process and args.logger == "wandb":
                        wandb.log(logs, step=step)
>>>>>>> 31c62ac905f5796de46a8b0a761dd76feea6b65a

                # Save the model
                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    if accelerator.is_main_process:
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.eval()

                        unwrapped_model.zero_grad()
                        states = [
                            unwrapped_model,
                            optimizer.state_dict(),
                            epoch,
                            step,
                            i
                        ]
                        if args.use_ema and self.config.model.ema:
                            states.append(ema_helper.state_dict())

                        torch.save(
                            states,
                            os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                        )
                        torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()

            accelerator.wait_for_everyone()
            # Save the model at the end of each epoch

            logs = {"loss at epoch": loss_at_epoch / len(train_loader),
                    "mse at epoch": mse_at_epoch / len(train_loader),
                    "kd at epoch": kd_at_epoch / len(train_loader),
                    "jac at epoch": jac_at_epoch / len(train_loader)}
            if accelerator.is_main_process and args.logger == "wandb":
<<<<<<< HEAD
                wandb.log(logs, epoch=epoch)
=======
                wandb.log(logs, step=step)
>>>>>>> 31c62ac905f5796de46a8b0a761dd76feea6b65a
            # Sampling for visualization at the end of each epoch
            if accelerator.is_main_process:

                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.eval()
                if ema_helper is not None:
                    ema_helper.store(unwrapped_model.parameters())
                    ema_helper.copy_to(unwrapped_model.parameters())

                with torch.no_grad():
                    n = config.sampling.batch_size
                    x = torch.randn(
                        n,
                        config.data.channels,
                        config.data.image_size,
                        config.data.image_size,
                        device=accelerator.device,
                    )
                    x = self.sample_image(x, model)
                    x = inverse_data_transform(config, x)
                    grid = tvu.make_grid(x)
                    tvu.save_image(grid, os.path.join(args.log_path, 'vis', 'epoch-{}.png'.format(epoch)))
                    wandb.log({"Epoch {} Sample".format(epoch): wandb.Image(grid)})

                if args.use_ema:
                    ema_helper.restore(unwrapped_model.parameters())

            #     if step > config.training.n_iters:
            #         progress_bar.close()
            #         accelerator.wait_for_everyone()
            #         accelerator.end_training()
            #         return
            # progress_bar.close()
            # accelerator.wait_for_everyone()
        accelerator.end_training()

    def sample(self):
        accelerator = self.accelerator
        model = self.model 
        model.to(accelerator.device)
        model.eval()
        
        if self.args.fid:
            self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            for i in tqdm(range(1)):
                import torch
                torch.manual_seed(i + self.args.seed)
                import random
                random.seed(i + self.args.seed)
                import numpy as np
                np.random.seed(i + self.args.seed)
                self.sample_sequence(model, index=i)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model):
        args = self.args
        accelerator = self.accelerator
        import torch
        torch.manual_seed(accelerator.process_index+args.seed)
        import random
        random.seed(accelerator.process_index+args.seed)
        import numpy as np
        np.random.seed(accelerator.process_index+args.seed)

        config = self.config

        total_n_samples = config.sampling.total_samples
        samples_for_each_process = int(total_n_samples / accelerator.num_processes / config.sampling.batch_size) * config.sampling.batch_size
        if accelerator.is_main_process:
            print(f"Samping {samples_for_each_process}x{accelerator.num_processes}"
                  f"={samples_for_each_process * accelerator.num_processes} (out of {config.sampling.total_samples}) images with {accelerator.num_processes} process(es)")

        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        n_rounds = (samples_for_each_process - img_id) // config.sampling.batch_size
        if accelerator.is_main_process:
            print(f"starting from image {img_id}")
        #os.makedirs(os.path.join(self.args.image_folder, '{}'.format(accelerator.process_index)), exist_ok=True)
        with torch.no_grad():
            for _ in tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation.", disable=not accelerator.is_main_process
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=accelerator.device,
                )

                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_sequence(self, model, index=0):
        config = self.config

        x = torch.randn(
            16,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        # print(f'x: {x[0,0,:10, :10]}')
        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.

        with torch.no_grad():
            # intermedias corresponds to predicted x0
            # x corresponds to xt
            # both are from t = 1000 to t = 0, with 101 values
            intermedias, x = self.sample_image(x, model, last=False)

        # pdb.set_trace()
        skip = self.num_timesteps // self.args.timesteps
        intermedias_x0_dict = {}
        intermedias_xt_dict = {}
        # seq = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] # do not need to save all steps
        # for i, t in enumerate(reversed(seq)):
        #     intermedias_x0_dict[t] = intermedias[int(i * skip)]
        #     intermedias_xt_dict[t] = x[int(i * skip)]

        for i, t in enumerate(range(1000, -1, -10)):
            intermedias_x0_dict[t] = intermedias[i]
            intermedias_xt_dict[t] = x[i]

        torch.save(intermedias_x0_dict, os.path.join(self.args.exp, f'intermediate_x0_{index}.pt'))
        torch.save(intermedias_xt_dict, os.path.join(self.args.exp, f'intermediate_xt_{index}.pt'))
        # x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps
            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def test(self):
        pass
