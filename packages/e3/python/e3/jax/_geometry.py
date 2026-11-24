# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, pure-JAX geometric operations."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def unit_displacement(sender: Any, receiver: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the unit displacement and distance from sender to receiver."""
    if sender.shape != receiver.shape or sender.ndim != 1:
        raise ValueError(
            f"sender and receiver have shapes {sender.shape} and {receiver.shape}, "
            "expected equal vectors"
        )
    displacement = receiver - sender
    squared_distance = jnp.sum(displacement * displacement)
    inverse_distance = squared_distance ** jnp.asarray(-0.5, squared_distance.dtype)
    return displacement * inverse_distance, squared_distance * inverse_distance


def unit_displacement_pbc(
    sender: Any,
    receiver: Any,
    shift: Any,
    lattice: Any,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return unit displacement and distance for a periodic image."""
    if sender.shape != receiver.shape or sender.ndim != 1:
        raise ValueError(
            f"sender and receiver have shapes {sender.shape} and {receiver.shape}, "
            "expected equal vectors"
        )
    dimension = sender.shape[0]
    if shift.shape != (dimension,):
        raise ValueError(f"shift has shape {shift.shape}, expected ({dimension},)")
    if lattice.shape != (dimension, dimension):
        raise ValueError(f"lattice has shape {lattice.shape}, expected ({dimension}, {dimension})")
    image = jnp.matmul(shift.astype(sender.dtype), lattice)
    return unit_displacement(sender, receiver + image)


def free_knot_radial(radius: Any, inverse_cutoff: Any, slope: Any) -> jnp.ndarray:
    """Evaluate ``(1-r/c)^2 (1+s r/c)`` for r <= c otherwise zero."""
    x = radius * inverse_cutoff
    return jnp.where(x <= 1, (1 - x) * (1 - x) * (1.0 + slope * x), 0.0)


__all__ = ["free_knot_radial", "unit_displacement", "unit_displacement_pbc"]
