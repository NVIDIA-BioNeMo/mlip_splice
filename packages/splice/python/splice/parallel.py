# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-process data parallelism: one process per GPU, ``jax.distributed``.

Topology (the only one we support): **one OS process drives exactly one GPU**.
The process is pinned to its GPU via ``CUDA_VISIBLE_DEVICES`` (so that GPU is
the process's device 0) and joins a global mesh over all processes' devices.
Params/optimizer state are **replicated** on every process; the batch is fed as
a global array whose leading "data" axis is sharded across processes, each
process contributing only its own sub-batch. The ``shard_map`` train step runs
the whole model per-process on local data and all-reduces gradients over the
"data" axis (NCCL). This is the *same* code from 2 GPUs on one box to 4xH100 to
multi-node — only the launcher's process count / coordinator change.

Why per-process-per-GPU rather than one process with N local devices: it is the
only topology that extends to multi-node (a process can't span nodes), and it
sidesteps an op that misbehaves on a non-default CUDA device within a single
process (each process sees its GPU as device 0).

Launch contract (set by the launcher, read by :func:`init_distributed`):
  SPLICE_NUM_PROCESSES, SPLICE_PROCESS_ID, SPLICE_COORDINATOR
  (+ CUDA_VISIBLE_DEVICES pinned to one GPU per process).
SLURM ``srun`` is also recognised (SLURM_NTASKS / SLURM_PROCID).
"""

from __future__ import annotations

import inspect
import os

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

DATA_AXIS = "data"


def _shard_map_check_kwarg() -> str | None:
    """The replication-check kwarg name on this JAX's ``shard_map`` — renamed
    ``check_rep`` -> ``check_vma`` across versions. Returns None if neither
    (so we just omit it). Keeps the train step portable across JAX versions."""
    params = inspect.signature(jax.shard_map).parameters
    for name in ("check_vma", "check_rep"):
        if name in params:
            return name
    return None


def shard_map(f, *, mesh: Mesh, in_specs, out_specs, check: bool = False):
    """Version-robust ``jax.shard_map`` wrapper. We disable the replication
    check by default — our outputs are made replicated explicitly via psum, and
    the checker can false-positive on Equinox-pytree outputs."""
    kwargs = {"mesh": mesh, "in_specs": in_specs, "out_specs": out_specs}
    check_kw = _shard_map_check_kwarg()
    if check_kw is not None:
        kwargs[check_kw] = check
    return jax.shard_map(f, **kwargs)


def init_distributed() -> dict:
    """Initialise ``jax.distributed`` from the launch environment.

    Returns ``{"num_processes", "process_id", "coordinator", "distributed"}``.
    A no-op (``distributed=False``) when only one process is requested, so the
    single-GPU path is unaffected.
    """
    num = int(os.environ.get("SPLICE_NUM_PROCESSES", os.environ.get("SLURM_NTASKS", "1")))
    pid = int(os.environ.get("SPLICE_PROCESS_ID", os.environ.get("SLURM_PROCID", "0")))
    coord = os.environ.get("SPLICE_COORDINATOR", "")

    if num <= 1:
        return {"num_processes": 1, "process_id": 0, "coordinator": "", "distributed": False}

    if not coord:
        raise RuntimeError(
            "SPLICE_NUM_PROCESSES>1 but SPLICE_COORDINATOR is unset "
            "(expected host:port of process 0)."
        )
    # Each process is pinned to one GPU via CUDA_VISIBLE_DEVICES, so it owns a
    # single local device (its device 0) -- do not pass local_device_ids.
    jax.distributed.initialize(coordinator_address=coord, num_processes=num, process_id=pid)
    if jax.local_device_count() != 1:
        raise RuntimeError(
            f"expected 1 local device per process, got {jax.local_device_count()}; "
            "pin each process to one GPU with CUDA_VISIBLE_DEVICES."
        )
    return {"num_processes": num, "process_id": pid, "coordinator": coord, "distributed": True}


def build_mesh(axis_name: str = DATA_AXIS) -> Mesh:
    """1-D mesh over **all** devices across all processes."""
    return Mesh(jax.devices(), (axis_name,))


def is_primary() -> bool:
    """True on process 0 only (does logging, checkpointing)."""
    return jax.process_index() == 0


def replicated(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, PartitionSpec())


def data_sharded(mesh: Mesh, axis_name: str = DATA_AXIS) -> NamedSharding:
    return NamedSharding(mesh, PartitionSpec(axis_name))


def replicate_tree(tree, mesh: Mesh):
    """Place a full copy of every array leaf on every device. Each process
    holds the whole tree on its local device; collectively this is one global
    replicated array. Non-array leaves (Equinox statics) pass through."""
    repl = replicated(mesh)
    return jax.tree.map(lambda x: jax.device_put(x, repl) if _is_array(x) else x, tree)


def commit_tree(tree, device=None):
    """Pin every array leaf of ``tree`` to this process's local device.

    Freshly initialised params/optimizer state are *uncommitted*: JAX has not
    bound them to a device, so ``jit`` sees their sharding as unspecified. The
    step's own outputs are committed, so the second call presents a different
    input sharding for identical shapes and XLA compiles the whole step a
    second time. Committing up front makes step 1 and step 2 the same
    executable. Non-array leaves (Equinox statics) pass through.
    """
    device = jax.local_devices()[0] if device is None else device
    return jax.tree.map(lambda x: jax.device_put(x, device) if _is_array(x) else x, tree)


def localize_replicated_tree(tree):
    """View this process's addressable replica without a device-to-host copy."""

    def local(x):
        if isinstance(x, jax.Array) and not x.is_fully_addressable:
            return x.addressable_data(0)
        return x

    return jax.tree.map(local, tree)


def place_local_batch(local_sub: dict, device=None) -> dict:
    """Host NumPy batch -> this process's local device (no extra D2D hop).

    Python metadata (``dummy_graph_id``) is omitted: the dummy graph is the
    last padded graph, recovered as ``nats.shape[-1] - 1`` inside the loss.
    Placement runs on the staging thread so H2D overlaps the previous step.
    """
    device = jax.local_devices()[0] if device is None else device
    out = {}
    for k, v in local_sub.items():
        if k == "dummy_graph_id":
            continue
        if _is_array(v):
            out[k] = jax.device_put(v, device)
        else:
            out[k] = v
    return out


def shard_batch(local_sub: dict, mesh: Mesh) -> dict:
    """Assemble the global sharded batch from this process's local sub-batch.

    ``local_sub`` is one complete, padded sub-batch built by THIS process for
    its own GPU (no device axis). Every process must pass an identically-shaped
    sub-batch (use the loader's fixed-bucket mode) so the global array is
    rectangular. The global leading "data" axis has one slice per process; this
    process owns the slice at its ``process_index``.
    """
    sh = data_sharded(mesh)
    n_proc = jax.process_count()
    local_dev = jax.local_devices()[0]
    out = {}
    for k, v in local_sub.items():
        if k == "dummy_graph_id":
            continue
        if not _is_array(v):
            out[k] = v
            continue
        host = np.asarray(v)
        local = jax.device_put(host[None], local_dev)
        out[k] = jax.make_array_from_single_device_arrays((n_proc, *host.shape), sh, [local])
    return out


def _is_array(x) -> bool:
    # Numpy scalars also have shape/dtype; they must stay host metadata.
    return isinstance(x, (np.ndarray, jax.Array))
