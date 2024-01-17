from dataclasses import dataclass
from typing import Callable, Optional

# model_type = "bert-base-cased"
model_type = "roberta-base"

# Glue tasks
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


@dataclass
class TaskConfig:
    train_seed: int = 0
    sample_seed: int = 0

    # Data hparams
    finetune_task_name: str = "stsb"
    max_seq_length: int = 32
    num_train_samples: Optional[int] = None

    # Finetune type hparams
    finetune_strategy: str = "full"  # "full" or "lora"
    finetune_filter: Callable = lambda _, v: len(v) == 2 and min(v) >= 768
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
