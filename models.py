from typing import Optional

import flax
import flax.linen as nn


class MatrixFactorization(nn.Module):
    outer_dims: int
    init_scale: float = 1e-2
    depth: int = 2
    inner_dims: Optional[int] = None

    def setup(self):
        assert self.depth >= 2, "Must have at least 2 factors"
        inner_dims = self.inner_dims if self.inner_dims else self.outer_dims
        layers = []
        layers.append(
            self.param(
                "w0",
                nn.initializers.orthogonal(scale=self.init_scale),
                (inner_dims, self.outer_dims),
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
                (self.outer_dims, inner_dims),
            )
        )
        self.layers = layers

    def __call__(self):
        x = self.layers[0]
        for w in self.layers[1:]:
            x = w @ x
        return x


class LoRA(nn.Module):
    flat_params_keys: list[str]
    outer_dims: int = 768
    init_scale: float = 1e-2
    depth: int = 2
    inner_dims: Optional[int] = None

    def setup(self):
        dmfs = {}
        for k in self.flat_params_keys:
            dmfs[k] = MatrixFactorization(
                outer_dims=self.outer_dims,
                init_scale=self.init_scale,
                depth=self.depth,
                inner_dims=self.inner_dims,
                name="_".join(k) + "_dmf",
            )
        self.dmfs = dmfs

    def compute_updates(self):
        return {k: v() for k, v in self.dmfs.items()}

    def __call__(self, model_params):
        updates = self.compute_updates()
        return flax.traverse_util.path_aware_map(
            lambda k, v: v + updates[k] if k in self.dmfs.keys() else v,
            model_params,
        )
