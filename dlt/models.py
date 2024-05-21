from typing import Optional, Tuple

import flax
import flax.linen as nn
import flax.traverse_util
import jax.numpy as jnp
from chex import ArrayTree
from transformers import (
    AutoConfig,
    FlaxAutoModel,
    FlaxAutoModelForSeq2SeqLM,
    FlaxAutoModelForSequenceClassification,
    PretrainedConfig,
)

from . import misc_utils, model_utils
from .configs import ModelType, TaskConfig


class MatrixFactorization(nn.Module):
    shape: Tuple[int, int]
    init_scale: float
    depth: int
    rank: Optional[int]

    def setup(self):
        assert self.depth >= 2, "Depth must be at least 2"
        set_width = self.rank if self.rank else min(self.shape)
        misc_utils.check_rank(set_width, self.shape)

        if self.depth == 2:
            init_fn = nn.initializers.normal(stddev=1)
            last_init_fn = nn.zeros_init()
        else:
            init_fn = nn.initializers.orthogonal(scale=self.init_scale)
            last_init_fn = init_fn

        layers = []
        layers.append(
            self.param(
                "W1",
                init_fn,
                (set_width, self.shape[1]),
            )
        )
        for i in range(2, self.depth):
            layers.append(
                self.param(
                    f"W{i}",
                    init_fn,
                    (set_width, set_width),
                )
            )
        layers.append(
            self.param(
                f"W{self.depth}",
                last_init_fn,
                (self.shape[0], set_width),
            )
        )
        self.layers = layers

    def __call__(self):
        x = self.layers[0]
        for w in self.layers[1:]:
            x = w @ x
        return x


class CompressedMatrixFactorization(nn.Module):
    shape: Tuple[int, int]
    init_scale: float
    depth: int
    rank: int

    def setup(self):
        self.left_factor = self.param(
            "left", nn.initializers.orthogonal(), (self.shape[0], self.rank)
        )
        self.right_factor = self.param(
            "right", nn.initializers.orthogonal(), (self.shape[1], self.rank)
        )
        self.mf = MatrixFactorization(
            (self.rank, self.rank), self.init_scale, self.depth, None
        )

    def __call__(self):
        return jnp.linalg.multi_dot([self.left_factor, self.mf(), self.right_factor.T])


class Lora(nn.Module):
    flat_params_shape_dict: dict
    init_scale: float
    depth: int
    rank: Optional[int]
    compressed: bool

    def setup(self):
        mfs = {}
        for flat_param_path, shape in self.flat_params_shape_dict.items():
            if self.compressed:
                assert (
                    self.rank is not None
                ), "rank must be specified for compressed LoRA"
                mf = CompressedMatrixFactorization(
                    shape=shape,
                    init_scale=self.init_scale,
                    depth=self.depth,
                    rank=self.rank,
                    name=flat_param_path,
                )
            else:
                mf = MatrixFactorization(
                    shape=shape,
                    init_scale=self.init_scale,
                    depth=self.depth,
                    rank=self.rank,
                    name=flat_param_path,
                )
            mfs[flat_param_path] = mf
        self.mfs = mfs

    def __call__(self):
        return {k: v() for k, v in self.mfs.items()}

    def adapt(self, model_params: ArrayTree) -> ArrayTree:
        updates = self()

        def f(k, v):
            flat_k = "/".join(k)
            if flat_k in updates.keys():
                return v + updates[flat_k]
            else:
                return v

        return flax.traverse_util.path_aware_map(
            f,
            model_params,  # type: ignore
        )


GLUE_TASK_TO_NUM_LABELS = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "qnli": 2,
    "qqp": 2,
    "rte": 2,
    "sst2": 2,
    "stsb": 1,
}


def create_pretrain_config_from_config(task_config: TaskConfig) -> PretrainedConfig:
    config = AutoConfig.from_pretrained(
        task_config.pretrain_model,
        num_labels=GLUE_TASK_TO_NUM_LABELS.get(task_config.finetune_task_name, 1),
    )
    return config


def create_pretrain_model_from_config(
    pretrain_model: ModelType,
    model_config: PretrainedConfig,
) -> FlaxAutoModel:
    "Creates transformer model from task config."

    model = (
        FlaxAutoModelForSequenceClassification
        if pretrain_model == ModelType.BERT
        else FlaxAutoModelForSeq2SeqLM
    ).from_pretrained(pretrain_model, config=model_config)

    return model


def create_lora_model_from_config(
    task_config: TaskConfig, model_params: ArrayTree
) -> Lora:
    "Creates LoRA model from task config and pretrain model parameters."

    filtered_flat_model_params_shape_dict = (
        model_utils.get_filtered_flat_params_shape_dict(
            model_params,
            task_config.lora_adapt_type,
        )
    )

    lora_model = Lora(
        flat_params_shape_dict=filtered_flat_model_params_shape_dict,  # type: ignore
        depth=task_config.lora_depth,
        init_scale=task_config.lora_init_scale,
        rank=task_config.lora_rank if not task_config.lora_compress else None,
        compressed=False,
    )

    return lora_model
