from dataclasses import dataclass
from typing import Optional

from data import task_to_keys


@dataclass
class ModelArguments:
    # pretrain_model: str = "bert-base-cased"
    pretrain_model = "roberta-base"


@dataclass
class DataArguments:
    finetune_task: str = "stsb"
    max_seq_length: int = 32
    max_train_samples: Optional[int] = 8

    def __post_init__(self):
        assert self.finetune_task in task_to_keys.keys()


@dataclass
class TrainArguments:
    finetune_strategy: str = "lora2"
    seed: int = 1
    num_train_epochs: int = 2000
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 4
    warmup_steps: int = 0
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    decay_ratio: float = 0.1
    log_steps: int = 200
    eval_steps: int = 200
