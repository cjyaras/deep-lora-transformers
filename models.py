from typing import Optional, Tuple

import flax
import flax.linen as nn
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
        "bert-base-cased",
        num_labels=num_labels,
        finetuning_task=task_config.finetune_task_name,
    )
    model = transformers.FlaxAutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", config=config
    )
    return model
