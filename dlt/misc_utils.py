from typing import Tuple

import numpy as np


def cosine_angle(U, V):
    "Computes the cosine subspace angle between two subspaces spanned by the columns of U and V."
    return np.linalg.norm(np.linalg.svd(U.T @ V, compute_uv=False), ord=np.inf)  # type: ignore


def check_rank(rank: int, shape: Tuple[int, int]):
    assert rank <= max(
        shape
    ), f"Rank {rank} must be no larger than outer dimensions {shape}"
