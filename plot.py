import matplotlib.pyplot as plt
import numpy as np

lora2 = np.genfromtxt("bert-base-cased-stsb-lora2.csv", delimiter=",")
lora2 = lora2[1:, 1:]
lora3 = np.genfromtxt("bert-base-cased-stsb-lora3.csv", delimiter=",")
lora3 = lora3[1:, 1:]

fig, ax = plt.subplots()
ax.plot(lora2[:, 0], lora2[:, 1], label="Depth=2")
ax.plot(lora3[:, 0], lora3[:, 1], label="Depth=3")
ax.set_title("Few Shot LoRA on STS-B", fontsize=16)
ax.set_xlabel("Iterations", fontsize=12)
ax.set_ylabel("Pearson Correlation", fontsize=12)
ax.legend(fontsize=12)
plt.savefig("bert-lora-stsb.png", dpi=500)
# plt.show()


# Hparams
# @dataclass
# class ModelArguments:
#     pretrain_model: str = "bert-base-cased"  # = "roberta-base"


# @dataclass
# class DataArguments:
#     finetune_task: str = "stsb"
#     max_seq_length: int = 32
#     max_train_samples: Optional[int] = 4

#     def __post_init__(self):
#         assert self.finetune_task in task_to_keys.keys()


# @dataclass
# class TrainArguments:
#     finetune_strategy: str = "lora2"
#     seed: int = 1
#     num_train_epochs: int = 2000
#     per_device_train_batch_size: int = 4
#     per_device_eval_batch_size: int = 32
#     warmup_steps: int = 0
#     learning_rate: float = 2e-5
#     weight_decay: float = 0.0
#     decay_ratio: float = 0.1
#     log_steps: int = 200
#     eval_steps: int = 200
