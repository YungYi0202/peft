from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, AutoTokenizer, TrainerCallback, \
    TrainerState, TrainerControl, Seq2SeqTrainer
from transformers import TrainingArguments

import os

# for LrRescheduleTrainer
from functools import partial
from torch.optim.lr_scheduler import LambdaLR

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


class LrRescheduleTrainer(Seq2SeqTrainer):
    def __init__(self, specified_epoch, total_epoch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add custom attributes here
        if specified_epoch is None:
            self.total_epoch = 1
            self.specified_epoch = 0
        else:
            self.total_epoch = total_epoch
            self.specified_epoch = specified_epoch
        
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        
        self.lr_scheduler = self.get_linear_schedule_with_warmup(
            optimizer=self.optimizer if optimizer is None else optimizer,
            num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return self.lr_scheduler

    def get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        """
        Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
        a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

        Args:
            optimizer ([`~torch.optim.Optimizer`]):
                The optimizer for which to schedule the learning rate.
            num_warmup_steps (`int`):
                The number of steps for the warmup phase.
            num_training_steps (`int`):
                The total number of training steps.
            last_epoch (`int`, *optional*, defaults to -1):
                The index of the last epoch when resuming training.

        Return:
            `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
        """

        lr_lambda = partial(
            self._get_linear_schedule_with_warmup_lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return LambdaLR(optimizer, lr_lambda, last_epoch)

    def _get_linear_schedule_with_warmup_lr_lambda(self, current_step: int, *, num_warmup_steps: int, num_training_steps: int):
        # The only difference
        # current_step += num_training_steps * input_arg['specified_epoch']
        current_step += num_training_steps * self.specified_epoch
        # num_training_steps *= input_arg['total_epoch']
        num_training_steps *= self.total_epoch

        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))



class FreezingCallback(TrainerCallback):
    def __init__(self, trainer, freeze_model, freeze_epoch=3):
        self.trainer = trainer
        self.freeze_model = freeze_model
        self.freeze_epoch = freeze_epoch
        self.current_step_idx = 0
        self.default_param_fix = {}
        self.name_list = []
        for name, param in self.freeze_model.named_parameters():
            self.name_list.append(name)
            self.default_param_fix[name] = param.requires_grad
        self.freeze_layers = int(len(self.default_param_fix.keys()) / freeze_epoch)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.epoch < self.freeze_epoch:
            release = self.name_list[-int(self.freeze_layers * state.epoch):]
            for name, param in self.freeze_model.named_parameters():
                if name in release:
                    param.requires_grad = self.default_param_fix[name]
                else:
                    param.requires_grad = False
        else:
            for name, param in self.freeze_model.named_parameters():
                param.requires_grad = self.default_param_fix[name]
        self.current_step_idx += 1

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        for name, param in self.trainer.model.named_parameters():
            param.requires_grad = True
