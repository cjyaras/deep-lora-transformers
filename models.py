from typing import Optional, Tuple

import flax
import flax.linen as nn
import jax
import transformers

import configs
import utils


class MatrixFactorization(nn.Module):
    shape: Tuple[int, int]
    init_scale: float
    depth: int
    rank: Optional[int]

    def setup(self):
        assert self.depth >= 2, "depth must be at least 2"
        set_rank = self.rank if self.rank else min(self.shape)
        assert set_rank <= min(
            self.shape
        ), f"rank {set_rank} must be smaller than outer dimensions {self.shape}"

        if self.depth == 2:
            init_fn = nn.initializers.normal(stddev=1)
            last_init_fn = nn.zeros_init()
        else:
            init_fn = nn.initializers.orthogonal(scale=self.init_scale)
            last_init_fn = init_fn

        layers = []
        layers.append(
            self.param(
                "w0",
                init_fn,
                (set_rank, self.shape[1]),
            )
        )
        for i in range(1, self.depth - 1):
            layers.append(
                self.param(
                    f"w{i}",
                    init_fn,
                    (set_rank, set_rank),
                )
            )
        layers.append(
            self.param(
                f"w{self.depth-1}",
                last_init_fn,
                (self.shape[0], set_rank),
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
            "right", nn.initializers.orthogonal(), (self.rank, self.shape[1])
        )
        self.mf = MatrixFactorization(
            (self.rank, self.rank), self.init_scale, self.depth, None
        )

    def __call__(self):
        return self.left_factor @ self.mf() @ self.right_factor


class LoRA(nn.Module):
    flat_params_shape_dict: dict
    init_scale: float
    depth: int
    rank: Optional[int]
    compressed: bool

    def setup(self):
        dmfs = {}
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
            dmfs[flat_param_path] = mf
        self.dmfs = dmfs

    def __call__(self):
        return {k: v() for k, v in self.dmfs.items()}

    def adapt(self, model_params):
        updates = self()

        def f(k, v):
            flat_k = "/".join(k)
            if flat_k in updates.keys():
                return v + updates[flat_k]
            else:
                return v

        return flax.traverse_util.path_aware_map(
            f,
            model_params,
        )


def create_pretrain_model_from_config(
    task_config: configs.TaskConfig, num_labels: int
) -> transformers.FlaxAutoModelForSequenceClassification:
    # Model
    config = transformers.AutoConfig.from_pretrained(
        task_config.pretrain_model,
        num_labels=num_labels,
        finetuning_task=task_config.finetune_task_name,
    )
    model = transformers.FlaxAutoModelForSequenceClassification.from_pretrained(
        task_config.pretrain_model, config=config
    )
    return model


def create_lora_model_from_config(
    task_config: configs.TaskConfig, model_params: flax.core.FrozenDict[str, jax.Array]
) -> LoRA:
    filtered_flat_model_params_shape_dict = utils.get_filtered_flat_params_shape_dict(
        model_params, task_config
    )

    lora_model = LoRA(
        flat_params_shape_dict=filtered_flat_model_params_shape_dict,  # type: ignore
        depth=task_config.lora_depth,
        init_scale=task_config.lora_init_scale,
        rank=task_config.lora_rank if not task_config.lora_compress else None,
        compressed=False,
    )

    return lora_model
