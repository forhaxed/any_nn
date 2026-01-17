"""
Helper class for train any model.
"""

from datetime import datetime
import json
from colorama import Fore, Style
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
import os
import logging
import torch
import time

class AnyTrainer:
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        self.batch_size = None
        self.epochs = None
        self.gradient_accumulation_steps = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.models = []
        self.non_trainable_models = []

        self.global_step = 0
        self.epochs_trained = 0
        self.steps_in_epoch = 0
        self.save_checkpoint_every_steps = 0
        self.eval_every_steps = 0
        self.max_grad_norm = None
        self.accelerator = None
        self.mixed_precision = "no"
        self.weight_dtype = torch.float32

    def init(self):
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        accelerator_project_config = ProjectConfiguration(project_dir=self.output_dir, logging_dir=os.path.join(self.output_dir, "logs/" + time_str))
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            log_with="tensorboard",
            project_config=accelerator_project_config
        )

        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        
        logger = get_logger(__name__)
        logger.info(self.accelerator.state, main_process_only=False)

        self.accelerator.print(f"{Fore.GREEN}AnyTrainer training started...{Style.RESET_ALL}\n")

        self.accelerator.print(f"{Fore.BLUE}Models Summary:{Style.RESET_ALL}")
        for i, model in enumerate(self.models):
            model_name = model.__class__.__name__
            num_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.accelerator.print(f" {model_name}:\n  Total Params: {num_params}\n  Trainable Params: {trainable_params}")
            if i < (len(self.models) - 1):
                self.accelerator.print("\n\n")

        if self.accelerator.is_main_process:
            if self.output_dir is not None:
                os.makedirs(self.output_dir, exist_ok=True)

        for i, model in enumerate(self.models):
            model.to(self.accelerator.device)
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.to(dtype=torch.float32)
                else:
                    param.data = param.to(dtype=self.weight_dtype)

            model.train()
            self.models[i] = self.accelerator.prepare(model)

        if self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer=self.optimizer, factor=1.0)

        self.optimizer = self.accelerator.prepare(self.optimizer)
        self.train_dataloader = self.accelerator.prepare(self.train_dataloader)
        self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader) if self.eval_dataloader is not None else None
        self.scheduler = self.accelerator.prepare(self.scheduler)

    def _save_state(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        
        self.accelerator.save_state(directory)

        metadata = {
            "global_step": self.global_step,
            "epochs_trained": self.epochs_trained,
            "steps_in_epoch": self.steps_in_epoch,
        }

        with open(os.path.join(directory, "trainer_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

    def save_state(self, directory: str):
        self._save_state(directory) # Internal save state

    def _load_state(self, directory: str):
        metadata_path = os.path.join(directory, "trainer_metadata.json")
        if os.path.isfile(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                self.global_step = metadata.get("global_step", 0)
                self.epochs_trained = metadata.get("epochs_trained", 0)
                self.steps_in_epoch = metadata.get("steps_in_epoch", 0)

            self.accelerator.load_state(directory)
            return True
        
        return False
    
    def load_state(self, directory: str):
        return self._load_state(directory)  # Internal load state

    def train_step(self, step, batch, device, weight_dtype):
        loss = 0.0
        return loss, {}
    
    def eval_begin(self, step):
        pass

    def eval_step(self, step, batch, device, weight_dtype):
        loss, loss_dict = self.train_step(step, batch, device, weight_dtype)
        return loss, loss_dict

    def eval_end(self, step):
        pass

    def gradient_sync(self, step):
        pass

    def train(self):
        if self.batch_size is None or self.epochs is None or self.optimizer is None or self.train_dataloader is None or self.gradient_accumulation_steps is None:
            raise ValueError("Trainer not properly initialized. Please call init() before training.")
        
        dataset_size = len(self.train_dataloader.dataset)
        effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        total_steps = (dataset_size // effective_batch_size) * self.epochs

        self.accelerator.print(f"\n{Fore.BLUE}Training Configuration:{Style.RESET_ALL}")
        self.accelerator.print(f" Device: {self.accelerator.device}")
        self.accelerator.print(f" Mixed Precision: {self.accelerator.mixed_precision}")
        self.accelerator.print(f" Dataset Size: {dataset_size}")
        self.accelerator.print(f" Total Epochs: {self.epochs}")
        self.accelerator.print(f" Batch Size: {self.batch_size}")
        self.accelerator.print(f" Gradient Accumulation Steps: {self.gradient_accumulation_steps}")
        self.accelerator.print(f" Effective Batch Size: {effective_batch_size}")
        self.accelerator.print(f" Total Training Steps: {total_steps}\n")

        for model in self.non_trainable_models:
            model.to(self.accelerator.device, dtype=self.weight_dtype)
            model.eval()
        
        if self.accelerator.is_main_process:
            self_simple_types = {}
            for attr_name in dir(self):
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, (int, float, str, bool)):
                    self_simple_types[attr_name] = attr_value
            self.accelerator.init_trackers("any_trainer", config=self_simple_types)

        progress_bar = tqdm(
            range(0, total_steps),
            initial=self.global_step,
            desc="Steps",
            disable=not self.accelerator.is_main_process,
        )
            
        from accelerate.data_loader import skip_first_batches

        for epoch in range(self.epochs_trained, self.epochs):
            train_loss_accs = {}

            # Skip already processed batches when resuming mid-epoch
            if self.steps_in_epoch > 0:
                active_dataloader = skip_first_batches(self.train_dataloader, self.steps_in_epoch * self.gradient_accumulation_steps)
                starting_step = self.steps_in_epoch * self.gradient_accumulation_steps
            else:
                active_dataloader = self.train_dataloader
                starting_step = 0
            
            for step, batch in enumerate(active_dataloader, start=starting_step):
                accumulate_context = self.accelerator.accumulate(self.models)
                with accumulate_context:
                    loss, loss_dict = self.train_step(self.global_step, batch, device=self.accelerator.device, weight_dtype=self.weight_dtype)
                    merged_loss_dict = loss_dict
                    merged_loss_dict["loss"] = loss

                    for key, value in merged_loss_dict.items():
                        if key not in train_loss_accs:
                            train_loss_accs[key] = 0.0
                        avg_value = self.accelerator.gather(value.repeat(self.batch_size)).mean()
                        train_loss_accs[key] += avg_value.item() / self.gradient_accumulation_steps

                    self.accelerator.backward(loss)
                    
                    # Calculate gradient norm before clipping
                    grad_norm = None
                    if self.accelerator.sync_gradients:
                        all_params = []
                        for group in self.optimizer.param_groups:
                            for p in group["params"]:
                                if p.grad is not None:
                                    all_params.append(p)
                        if all_params:
                            grad_norm = self.accelerator.clip_grad_norm_(all_params, float('inf') if self.max_grad_norm is None else self.max_grad_norm)
                    
                    self.optimizer.step()
                    if self.accelerator.sync_gradients and self.scheduler is not None:
                        self.scheduler.step()

                    self.optimizer.zero_grad()

                    logs = {"step_loss": loss.detach().item()}
                    progress_bar.set_postfix(**logs)

                if self.accelerator.sync_gradients:
                    self.gradient_sync(self.global_step)
                    progress_bar.update(1)

                    effective_lr = self.scheduler.get_last_lr()[0]
                    log_dict = {"train/lr": effective_lr}
                    if grad_norm is not None:
                        log_dict["train/grad_norm"] = grad_norm

                    # Calculate average magnitude of parameters for each model
                    for model_idx, model in enumerate(self.models):
                        unwrapped_model = self.accelerator.unwrap_model(model)
                        total_magnitude = 0.0
                        total_params = 0
                        for param in unwrapped_model.parameters():
                            if param.requires_grad:
                                total_magnitude += param.data.abs().sum().item()
                                total_params += param.numel()
                        if total_params > 0:
                            avg_magnitude = total_magnitude / total_params
                            model_name = unwrapped_model.__class__.__name__
                            log_dict[f"train/avg_magnitude/{model_name}"] = avg_magnitude

                    for key, value in train_loss_accs.items():
                        log_dict[f"train/{key}"] = value

                    self.accelerator.log(log_dict, step=self.global_step)
                    train_loss_accs = {}

                    if self.accelerator.is_main_process:
                        should_save_checkpoint_file = os.path.join(self.output_dir, "save_checkpoint.lock")
                        if os.path.isfile(should_save_checkpoint_file):
                            os.remove(should_save_checkpoint_file)
                            should_save_checkpoint_file = True
                            print(f"Detected save checkpoint lock file. Saving checkpoint at step {self.global_step}...")
                        else:
                            should_save_checkpoint_file = False

                        should_eval_file = os.path.join(self.output_dir, "do_eval.lock")
                        if os.path.isfile(should_eval_file):
                            os.remove(should_eval_file)
                            should_eval_file = True
                            print(f"Detected do eval lock file. Running evaluation at step {self.global_step}...")
                        else:
                            should_eval_file = False

                        if should_save_checkpoint_file or (self.save_checkpoint_every_steps > 0 and self.global_step % self.save_checkpoint_every_steps == 0):
                            checkpoint_dir = os.path.join(self.output_dir, "checkpoints", f"step_{self.global_step}")
                            self.save_state(checkpoint_dir)
                            self.accelerator.print(f"Saved checkpoint to {checkpoint_dir}")

                        if should_eval_file or (self.eval_every_steps > 0 and self.global_step % self.eval_every_steps == 0):
                            self.eval_begin(self.global_step)

                            if self.eval_dataloader is not None:
                                self.accelerator.print(f"\nStarting evaluation at step {self.global_step}...")
                                eval_loss_accs = {}
                                num_eval_steps = 0

                                # Store wrapped models (copy the list, not reference)
                                original_models = list(self.models)

                                # Unwrap models for evaluation
                                for i, model in enumerate(original_models):
                                    self.models[i] = self.accelerator.unwrap_model(model)
                                    self.models[i].eval()

                                with torch.no_grad():
                                    for eval_batch in self.eval_dataloader:
                                        eval_loss, eval_loss_dict = self.eval_step(self.global_step, eval_batch, device=self.accelerator.device, weight_dtype=self.weight_dtype)
                                        merged_eval_loss_dict = eval_loss_dict
                                        if eval_loss is not None:
                                            merged_eval_loss_dict["loss"] = eval_loss

                                        for key, value in merged_eval_loss_dict.items():
                                            if key not in eval_loss_accs:
                                                eval_loss_accs[key] = 0.0
                                            eval_loss_accs[key] += value.item() if isinstance(value, torch.Tensor) else value

                                        num_eval_steps += 1
                                
                                    for key in eval_loss_accs:
                                        eval_loss_accs[key] /= num_eval_steps

                                    eval_log_dict = {}
                                    for key, value in eval_loss_accs.items():
                                        eval_log_dict[f"eval/{key}"] = value

                                # Restore original models
                                self.models = original_models
                                for i, model in enumerate(self.models):
                                    self.models[i].train()

                                self.accelerator.log(eval_log_dict, step=self.global_step)
                                self.accelerator.print(f"Evaluation results at step {self.global_step}: {eval_log_dict}\n")

                            self.eval_end(self.global_step)

                    self.steps_in_epoch += 1
                    self.global_step += 1

                    pause_file = os.path.join(self.output_dir, "pause.lock")

                    should_pause = torch.tensor(
                        [1 if (self.accelerator.is_main_process and os.path.isfile(pause_file)) else 0],
                        device="cpu"
                    )
                    should_pause = self.accelerator.reduce(should_pause, reduction="sum").item() > 0

                    if should_pause:
                        self.accelerator.wait_for_everyone()

                        self.accelerator.print(f"{Fore.YELLOW}Pausing training at step {self.global_step} due to presence of pause.lock file.{Style.RESET_ALL}")
                        
                        for model in self.models:
                            model.to("cpu")
                        
                        for model in self.non_trainable_models:
                            model.to("cpu")

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        while True:
                            self.accelerator.wait_for_everyone()
                            
                            still_paused = torch.tensor(
                                [1 if (self.accelerator.is_main_process and os.path.isfile(pause_file)) else 0],
                                device="cpu"
                            )
                            still_paused = self.accelerator.reduce(still_paused, reduction="sum").item() > 0
                            
                            if not still_paused:
                                break
                            time.sleep(1.0)

                        self.accelerator.print(f"{Fore.GREEN}Resuming training at step {self.global_step}.{Style.RESET_ALL}")
                        
                        for model in self.models:
                            model.to(self.accelerator.device)

                        for model in self.non_trainable_models:
                            model.to(self.accelerator.device)

                        self.accelerator.wait_for_everyone()

            self.steps_in_epoch = 0  # Reset for next epoch
            self.epochs_trained += 1

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()

        if self.accelerator.is_main_process:
            self.accelerator.print(f"{Fore.GREEN}Training complete!{Style.RESET_ALL}\n")
            checkpoint_dir = os.path.join(self.output_dir, "checkpoints", f"step_{self.global_step}_final")
            self.save_state(checkpoint_dir)