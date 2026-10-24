"""Exact, recursive state and RNG helpers for resumable GRPO runs."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from typing import Any


def _update_digest(digest: Any, value: Any) -> None:
    """Hash nested optimizer/scaler/RNG values without repr truncation."""

    if value is None:
        digest.update(b"none;")
        return
    if isinstance(value, bool):
        digest.update(f"bool:{value};".encode())
        return
    if isinstance(value, int):
        digest.update(f"int:{value};".encode())
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non_finite_state_value")
        digest.update(f"float:{value.hex()};".encode())
        return
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"str:")
        digest.update(str(len(encoded)).encode())
        digest.update(b":")
        digest.update(encoded)
        digest.update(b";")
        return
    if isinstance(value, bytes | bytearray):
        digest.update(b"bytes:")
        digest.update(str(len(value)).encode())
        digest.update(b":")
        digest.update(bytes(value))
        digest.update(b";")
        return

    # Torch is intentionally lazy so config/CPU validation never imports it.
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "shape"):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor:")
        digest.update(str(tensor.dtype).encode())
        digest.update(repr(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
        digest.update(b";")
        return

    if hasattr(value, "tobytes") and hasattr(value, "dtype") and hasattr(value, "shape"):
        digest.update(b"array:")
        digest.update(str(value.dtype).encode())
        digest.update(repr(tuple(value.shape)).encode())
        digest.update(value.tobytes())
        digest.update(b";")
        return

    if isinstance(value, Mapping):
        digest.update(b"map{")
        for key in sorted(value, key=lambda item: str(item)):
            _update_digest(digest, str(key))
            _update_digest(digest, value[key])
        digest.update(b"};")
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        digest.update(b"seq[")
        for item in value:
            _update_digest(digest, item)
        digest.update(b"];")
        return
    if isinstance(value, set | frozenset):
        digest.update(b"set{")
        for item in sorted(value, key=lambda item: repr(item)):
            _update_digest(digest, item)
        digest.update(b"};")
        return
    raise TypeError(f"unsupported_state_value:{type(value).__name__}")


def recursive_fingerprint(value: Any) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, value)
    return digest.hexdigest()


def recursive_equal(left: Any, right: Any) -> bool:
    """Compare every nested tensor/array/scalar value exactly."""

    if left is right:
        return True
    if type(left) is not type(right):
        # Torch CPU/CUDA state tensors can differ in device while representing
        # the same bytes; compare them after moving to CPU below.
        if not (
            hasattr(left, "detach")
            and hasattr(right, "detach")
            and hasattr(left, "shape")
            and hasattr(right, "shape")
        ):
            return False
    if hasattr(left, "detach") and hasattr(right, "detach"):
        return bool((left.detach().cpu() == right.detach().cpu()).all().item())
    if hasattr(left, "shape") and hasattr(right, "shape") and hasattr(left, "tobytes"):
        return (
            left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and left.tobytes() == right.tobytes()
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            recursive_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, Sequence) and isinstance(right, Sequence) and not isinstance(
        left, str | bytes | bytearray
    ):
        return len(left) == len(right) and all(
            recursive_equal(a, b) for a, b in zip(left, right, strict=False)
        )
    if isinstance(left, set | frozenset) and isinstance(right, set | frozenset):
        return left == right
    return left == right


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate()}
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        state["numpy"] = None
    try:
        import torch

        state["torch_cpu"] = torch.get_rng_state()
        state["torch_cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    except ImportError:
        state["torch_cpu"] = None
        state["torch_cuda"] = []
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        import numpy as np

        np.random.set_state(numpy_state)
    torch_state = state.get("torch_cpu")
    if torch_state is not None:
        import torch

        torch.set_rng_state(torch_state.cpu())
        cuda_states = state.get("torch_cuda") or []
        if cuda_states and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
