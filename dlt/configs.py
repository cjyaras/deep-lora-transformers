from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional, Tuple, Union

from dataclasses_json import dataclass_json


class ExtendedEnum(StrEnum):

    @classmethod
    def values(cls):
        return [e.value for e in cls]


class ModelType(ExtendedEnum):
    BERT = "google-bert/bert-base-cased"
    T5 = "google-t5/t5-base"


class TaskType(ExtendedEnum):
    GLUE = "glue"
    NLG = "nlg"


class LoraAdaptType(ExtendedEnum):
    ONLY_QUERY_VALUE = "only-query-value"
    ATTENTION_MLP = "attention-mlp"
    ALL_DENSE = "all-dense"


class GlueTaskName(ExtendedEnum):
    COLA = "cola"
    MNLI = "mnli"
    MRPC = "mrpc"
    QNLI = "qnli"
    QQP = "qqp"
    RTE = "rte"
    SST2 = "sst2"
    STSB = "stsb"


class NLGTaskName(ExtendedEnum):
    E2E_NLG = "e2e_nlg"


@dataclass_json
@dataclass
class TaskConfig:
    identifier: Optional[str] = None

    # Data hparams
    task_type: TaskType = TaskType.GLUE
    finetune_task_name: Union[GlueTaskName, NLGTaskName] = GlueTaskName.STSB
    max_seq_length: Union[int, Tuple[int, int]] = 128
    num_train_samples: Optional[int] = None

    # Model hparams
    pretrain_model: ModelType = ModelType.BERT

    # Lora hparams
    lora_adapt_type: LoraAdaptType = LoraAdaptType.ONLY_QUERY_VALUE
    lora_depth: int = 3
    lora_init_scale: float = 1e-3
    lora_rank: Optional[int] = None
    lora_compress: bool = False
    lora_random_factors: bool = False
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
    save_dir: str = "checkpoints"
