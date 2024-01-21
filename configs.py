from dataclasses import dataclass
from typing import Callable, Optional

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
    max_seq_length: Optional[int] = 32
    num_train_samples: Optional[int] = None

    # Model hparams
    pretrain_model: str = "bert-base-cased"

    # Lora hparams
    use_lora: bool = True
    lora_adapt_filter: Callable = lambda _, shape: len(shape) == 2 and min(shape) >= 768
    lora_depth: int = 3
    lora_init_scale: float = 1e-3
    lora_rank: Optional[int] = None

    # Training hparams
    num_train_steps: int = 400
    train_batch_size: int = 4
    eval_batch_size: int = 32
    num_warmup_steps: int = 0
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    decay_ratio: float = 0.1

    # Logging hparams
    log_steps: int = 10
    eval_steps: int = 10

    def __post_init__(self):
        assert self.finetune_task_name in task_to_keys.keys()
