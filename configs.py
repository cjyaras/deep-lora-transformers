from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from dataclasses_json import dataclass_json

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


class LoraAdaptType(str, Enum):
    only_query_value = "only_query_value"
    all_dense = "all_dense"


@dataclass_json
@dataclass
class TaskConfig:
    train_seed: int = 0
    sample_seed: int = 0

    # Data hparams
    finetune_task_name: str = "stsb"
    max_seq_length: Optional[int] = 32
    num_train_samples: Optional[int] = 4

    # Model hparams
    pretrain_model: str = "bert-base-cased"

    # Lora hparams
    lora_adapt_type: LoraAdaptType = LoraAdaptType.only_query_value
    lora_depth: int = 3
    lora_init_scale: float = 1e-3
    lora_rank: Optional[int] = None
    lora_alpha: int = 1

    # Training hparams
    num_train_steps: int = 100
    train_batch_size: int = 4
    eval_batch_size: int = 32
    num_warmup_steps: int = 0
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    decay_ratio: float = 0.1

    # Logging hparams
    log_eval_steps: int = 20
    save_step_points: list = field(default_factory=list)

    def __post_init__(self):
        assert self.finetune_task_name in task_to_keys.keys()
