from typing import Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import transformers

import configs


class MatrixFactorization(nn.Module):
    outer_dims: int | Tuple[int, int]
    init_scale: float = 1e-2
    depth: int = 2
    inner_dims: Optional[int] = None

    def setup(self):
        outer_dims = (
            (self.outer_dims, self.outer_dims)
            if isinstance(self.outer_dims, int)
            else self.outer_dims
        )
        inner_dims = self.inner_dims if self.inner_dims else min(outer_dims)
        assert inner_dims <= min(
            outer_dims
        ), f"inner_dims {inner_dims} must be smaller than outer_dims {outer_dims}"
        layers = []
        if self.depth == 1:
            layers.append(
                self.param(
                    "w",
                    nn.initializers.orthogonal(scale=self.init_scale),
                    outer_dims,
                )
            )
            self.layers = layers
            return
        else:
            layers.append(
                self.param(
                    "w0",
                    nn.initializers.orthogonal(scale=self.init_scale),
                    (inner_dims, outer_dims[1]),
                )
            )
            for i in range(1, self.depth - 1):
                layers.append(
                    self.param(
                        f"w{i}",
                        nn.initializers.orthogonal(scale=self.init_scale),
                        (inner_dims, inner_dims),
                    )
                )
            layers.append(
                self.param(
                    f"w{self.depth-1}",
                    nn.initializers.orthogonal(scale=self.init_scale),
                    (outer_dims[0], inner_dims),
                )
            )
        self.layers = layers

    def __call__(self):
        if self.depth == 1:
            return self.layers[0]
        else:
            x = self.layers[0]
            for w in self.layers[1:]:
                x = w @ x
            return x


class LoRA(nn.Module):
    flat_params_shape_dict: dict
    init_scale: float = 1e-2
    depth: int = 2
    inner_dims: Optional[int] = None

    def setup(self):
        dmfs = {}
        for param, shape in self.flat_params_shape_dict.items():
            dmfs[param] = MatrixFactorization(
                outer_dims=shape,
                init_scale=self.init_scale,
                depth=self.depth,
                inner_dims=self.inner_dims,
                name=param,
            )
        self.dmfs = dmfs

    def compute_updates(self):
        return {k: v() for k, v in self.dmfs.items()}

    def __call__(self, model_params):
        updates = self.compute_updates()

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


def create_pretrain_model_from_config(task_config: configs.TaskConfig, num_labels: int):
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
):
    flat_model_params = flax.traverse_util.flatten_dict(model_params, sep="/")
    flat_model_params_shape_dict = jax.tree_util.tree_map(jnp.shape, flat_model_params)
    filtered_flat_model_params_shape_dict = {
        k: v
        for k, v in flat_model_params_shape_dict.items()
        if task_config.finetune_filter(k, v)
    }

    lora_model = LoRA(
        flat_params_shape_dict=filtered_flat_model_params_shape_dict,  # type: ignore
        depth=task_config.lora_depth,
        init_scale=task_config.lora_init_scale,
        inner_dims=task_config.lora_rank,
    )

    return lora_model
