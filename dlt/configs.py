from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional, Tuple, Union

from dataclasses_json import dataclass_json


class ModelType(StrEnum):
    BERT = "bert-base-cased"
    BART = "facebook/bart-base"


class TaskType(StrEnum):
    GLUE = "glue"
    SUMMARIZATION = "summarization"


class LoraAdaptType(StrEnum):
    ONLY_QUERY_VALUE = "only-query-value"
    ATTENTION_MLP = "attention-mlp"
    ALL_DENSE = "all-dense"


class GlueTaskName(StrEnum):
    COLA = "cola"
    MNLI = "mnli"
    MRPC = "mrpc"
    QNLI = "qnli"
    QQP = "qqp"
    RTE = "rte"
    SST2 = "sst2"
    STSB = "stsb"


class SummarizationTaskName(StrEnum):
    CNN_DAILYMAIL = "cnn_dailymail"
    SAMSUM = "samsum"
    XSUM = "xsum"


@dataclass_json
@dataclass
class TaskConfig:
    identifier: Optional[str] = None

    # Data hparams
    task_type: TaskType = TaskType.GLUE
    finetune_task_name: Union[GlueTaskName, SummarizationTaskName] = GlueTaskName.STSB
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
    save_dir: str = "../checkpoints"

    def __post_init__(self):
        if self.task_type == TaskType.GLUE:
            assert isinstance(self.finetune_task_name, GlueTaskName)
            assert self.pretrain_model == ModelType.BERT
            assert isinstance(self.max_seq_length, int)
        elif self.task_type == TaskType.SUMMARIZATION:
            assert isinstance(self.finetune_task_name, SummarizationTaskName)
            assert self.pretrain_model == ModelType.BART
            assert isinstance(self.max_seq_length, Tuple)
        else:
            raise ValueError(f"Invalid task_type: {self.task_type}")
