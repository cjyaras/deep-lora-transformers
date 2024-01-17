from dataclasses import dataclass
from typing import Optional

from data import task_to_keys


@dataclass
class TaskConfig:
    seed: int = 0

    # Data hparams
    finetune_task_name: str = "stsb"
    max_seq_length: int = 32
    num_train_samples: Optional[int] = 4

    # Finetune type hparams
    finetune_strategy: str = "full"  # "full" or "lora"
    lora_depth: int = 2
    lora_init_scale: float = 1e-2
    lora_rank: int = 768

    # Training hparams
    num_train_epochs: int = 2000
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 32
    warmup_steps: int = 0
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    decay_ratio: float = 0.1

    # Logging hparams
    log_steps: int = 200
    eval_steps: int = 200

    def __post_init__(self):
        assert self.finetune_task_name in task_to_keys.keys()
