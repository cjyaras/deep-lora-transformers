import json
import os

import datasets
import matplotlib.pyplot as plt
import numpy as np

import configs
import models
from finetune import finetune

task_config = configs.TaskConfig()

task_config.task_type = configs.TaskType.SUMMARIZATION
task_config.pretrain_model = configs.ModelType.BART
task_config.finetune_task_name = configs.SummarizationTaskName.XSUM
task_config.max_seq_length = (512, 128)
task_config.lora_adapt_type = configs.LoraAdaptType.ONLY_QUERY_VALUE


# task_config.lora_adapt_type = configs.LoraAdaptType.ATTENTION_MLP
task_config.lora_init_scale = 1e-3
task_config.num_train_steps = 2000
task_config.train_batch_size = 16
# task_config.max_seq_length = 128
# task_config.log_eval_steps = 100
task_config.log_eval_steps = 1
task_config.learning_rate = 1e-4
task_config.decay_ratio = 0.1
task_config.save_step_points = [
    0,
    1,
    10,
    20,
    30,
    100,
    500,
    1000,
    1500,
    task_config.num_train_steps,
]
task_config.identifier = "intro"

finetune(task_config)
