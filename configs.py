from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from dataclasses_json import dataclass_json


class LoraAdaptType(str, Enum):
    only_query_value = "only-query-value"
    attention_mlp = "attention-mlp"
    all_dense = "all-dense"


@dataclass_json
@dataclass
class TaskConfig:
    identifier: Optional[str] = None

    # Data hparams
    finetune_task_name: str = "stsb"
    max_seq_length: Optional[int] = None
    num_train_samples: Optional[int] = None

    # Model hparams
    pretrain_model: str = "bert-base-cased"

    # Lora hparams
    lora_adapt_type: LoraAdaptType = LoraAdaptType.only_query_value
    lora_depth: int = 3
    lora_init_scale: float = 1
    lora_rank: Optional[int] = None
    lora_compress: bool = False
    lora_gamma: float = 0

    # Training hparams
    num_train_steps: int = 200
    train_batch_size: int = 1
    eval_batch_size: int = 32
    num_warmup_steps: int = 0
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    decay_ratio: float = 0.1

    # Logging hparams
    log_eval_steps: int = 200
    save_step_points: list = field(default_factory=list)
    save_dir: str = "experiments"
