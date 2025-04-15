# handles everything for hf Trainer
from transformers.trainer import (
    Trainer, 
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
)
from transformers.integrations import WandbCallback, rewrite_logs
from transformers.trainer_callback import ProgressCallback, PrinterCallback, CallbackHandler
from transformers.utils.import_utils import is_datasets_available
from transformers.trainer_utils import set_seed
from torch.utils.data import DataLoader
import torch
from functools import partial

if is_datasets_available():
    import datasets

def update_logs_with_losses(logs, state):
    losses = getattr(state, 'losses_for_record', None)
    if losses is not None:
        logs.update(losses)
    else:
        print(f"You used a `WithLosses` Callback but no sub losses are reported!")
    return logs


class WandbCallbackWithLosses(WandbCallback):
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            logs = update_logs_with_losses(logs, state)
            logs = rewrite_logs(logs)
            self._wandb.log({**logs, "train/global_step": state.global_step})

class ProgressCallbackWithLosses(ProgressCallback):
    # hack some necessary callbacks is enough
    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        if not hasattr(state, "losses_for_record"):
            state.losses_for_record = dict()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and self.training_bar is not None:
            logs = update_logs_with_losses(logs, state)
            _ = logs.pop("total_flos", None)
            self.training_bar.write(str(logs))

class PrinterCallbackWithLosses(PrinterCallback):
    # hack some necessary callbacks is enough
    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        if not hasattr(state, "losses_for_record"):
            state.losses_for_record = dict()

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = update_logs_with_losses(logs, state)
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            print(logs)

REPLACE_MAPPING = {
    WandbCallback: WandbCallbackWithLosses,
    ProgressCallback: ProgressCallbackWithLosses,
    PrinterCallback: PrinterCallbackWithLosses,
}

def hack_callbacks_and_replace(callback_handler: CallbackHandler):
    for i, callback in enumerate(callback_handler.callbacks):
        for ins in REPLACE_MAPPING.keys():
            if isinstance(callback, ins):
                callback_handler.callbacks[i] = REPLACE_MAPPING[ins]()
                break
    return callback_handler


class ExtractiveTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state.losses_for_record = dict()
        self.callback_handler = hack_callbacks_and_replace(self.callback_handler)

    # override for logging outputs (`LongformerExtractiveOutput`) to wandb 
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        overwrite the trainer to collect sub-loss
        """
        # remove label_smoother part 1
        # if self.label_smoother is not None and "labels" in inputs:
        #     labels = inputs.pop("labels")
        # else:
        #     labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # if labels is not None:
        #     if is_peft_available() and isinstance(model, PeftModel):
        #         model_name = unwrap_model(model.base_model)._get_name()
        #     else:
        #         model_name = unwrap_model(model)._get_name()
        #     if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
        #         loss = self.label_smoother(outputs, labels, shift_labels=True)
        #     else:
        #         loss = self.label_smoother(outputs, labels)
        # else:
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # collect all keys end with _loss; divided by accumulation steps; gather them, mean them then log as scalars
        if self.state.global_step % self.args.logging_steps == 0:
            for k, v in outputs.items():
                if k == 'loss':
                    k = 'debug_loss'  # should be the same with `loss`
                if k.endswith('loss'):
                    # any log call backs will retrieve the loss from state.losses_for_record
                    self.state.losses_for_record[k] = self._nested_gather(v).mean().item()
                if k.lower().endswith('map') or k.lower().startswith('recall'):  # acc does not need gather; they are scalars
                    self.state.losses_for_record[k] = v
                # add dict unpacking utils
                if k == 'unpack_log':
                    assert isinstance(v, dict), "unpack_log should be a dict"
                    for kk, vv in v.items():
                        if isinstance(vv, torch.Tensor):
                            vv = vv.item()
                        self.state.losses_for_record[kk] = vv

        return (loss, outputs) if return_outputs else loss
    
    # seperate lr for longformer & resnet
    def create_optimizer(self):
        """
        setup optimizer with different learning rate for different modules
        about weight_decay: layer_norm / pos_embed / bias are not decayed; resnet gets full decay
        removed segamake dependency as well 
        """
        # if not found visual_lr in the self.args, we skip
        if not hasattr(self.args, "visual_lr"):
            return super().create_optimizer()
        
        opt_model = self.model
        
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.visual_lr is not None or self.args.visual_weight_decay is not None:
                visual_lr = self.args.visual_lr if self.args.visual_lr is not None else self.args.learning_rate
                visual_weight_decay = self.args.visual_weight_decay if self.args.visual_weight_decay is not None else self.args.weight_decay
                visual_parameters = [name for name, _ in opt_model.named_parameters() if "visual_encoder" in name]
                
                optimizer_grouped_parameters = [
                    {  # normal decay parameters
                        "params": [p for n, p in opt_model.named_parameters() if (
                            n in decay_parameters and 
                            n not in visual_parameters and
                            p.requires_grad
                        )],
                        "weight_decay": self.args.weight_decay,
                    }, 
                    {  # normal no decay parameters
                        "params": [p for n, p in opt_model.named_parameters() if (
                            n not in decay_parameters and 
                            n not in visual_parameters and
                            p.requires_grad
                        )],
                        "weight_decay": 0.0,
                    }, 
                    {  # decay visual encoder parameters
                        "params": [p for n, p in opt_model.named_parameters() if (
                            n in visual_parameters and 
                            n in decay_parameters and
                            p.requires_grad
                        )],
                        "weight_decay": visual_weight_decay,
                        "lr": visual_lr,
                    }, 
                    { # no decay visual encoder parameters
                        "params": [p for n, p in opt_model.named_parameters() if (
                            n in visual_parameters and 
                            n not in decay_parameters and
                            p.requires_grad
                        )],
                        "weight_decay": 0.0,
                        "lr": visual_lr,
                    }
                ]  # complete optimizer group
            else:
                # normal optimizer branch
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            
            # collect optimizer factory and kwargs
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            # skip ShardedDDPOption
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            # skip adam8bit compatibility
        
        return self.optimizer
            
    
    # run once before training begins; wait to be tested with 25 (20, 23)
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if self.args.lr_scheduler_type == "constant" and self.args.lr_decay_epochs is not None:
            if self.lr_scheduler is None:
                # print(f"Using constant learning rate with decay at epochs {self.args.lr_decay_epochs}")
                def lr_lambda(current_step: int):
                    # Calculate the current epoch; self.args.lr_decay_epochs is a list of epochs that we want to decay the learning rate
                    # by the factor of self.args.lr_decay_ratio
                    epoch = current_step // (num_training_steps // self.args.num_train_epochs) + 1
                    lr_factor = 1.0
                    for decay_epoch in self.args.lr_decay_epochs:
                        if epoch >= decay_epoch:
                            lr_factor *= self.args.lr_decay_ratio
                    return lr_factor
                
                self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer if optimizer is None else optimizer, lr_lambda)
                self._created_lr_scheduler = True
                return self.lr_scheduler
        else:
            # print(f"Using {self.args.lr_scheduler_type} learning rate scheduler")
            return super().create_scheduler(num_training_steps, optimizer)

    def get_train_dataloader(self):
        # my_seed_worker in get_train_dataloader should depend on 
        # 1) worker_id,
        # 2) epoch, (depending if persistent dataloader is used, we may need to set seed for each epoch)
        # 3) base_seed (varies for each main process)
        # leave enough space to prevent seed collision; but in this way, how do we decide a good seed? with a v1 model?
        base_seed = self.args.base_seed   # a special field for sampling deterministic
        
        if base_seed is None:
            return super().get_train_dataloader()

        print(f"Using base_seed {base_seed} for deterministic training.")
        
        def my_seed_worker(worker_id):  # better though.
            # although torch.initial_seed() already considers worker_id offset, we want it completely deterministic
            seed = base_seed if base_seed is not None else torch.initial_seed() % 2**32  
            print(f"Setting seed {seed + worker_id} for worker {worker_id} at Process #{torch.multiprocessing.current_process().pid}")
            set_seed(seed + worker_id)
        
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": True,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = my_seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
    
    # on 03/31: added for multi-turn evaling and manual seeding; reporting the mean and std.
    def get_eval_dataloader(self, eval_dataset = None, base_seed: int = None):
        """
        seed has be different for each main process and each worker.
        """
        if base_seed is None:
            return super().get_eval_dataloader(eval_dataset)

        def my_seed_worker(worker_id):
            # purpose: maintain maximum reproducibility by set:
            # 1) fixed seed for each worker under the same CPU process
            # 2) different seed to different CPU process
            seed = base_seed if base_seed is not None else torch.initial_seed() % 2**32  # torch.initial_seed() already considers worker_id offset
            print(f"Setting seed {seed + worker_id} for worker {worker_id} at Process #{torch.multiprocessing.current_process().pid}")
            set_seed(seed + worker_id)
        
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = my_seed_worker
            # add seed_worker to dataloader_params

        return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))
                