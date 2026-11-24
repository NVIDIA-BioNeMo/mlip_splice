# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from loom._src.domain_index.dag import make_index_map_dag, make_preimages
from loom._src.domain_index.graph import DomainGraph
from loom._src.domain_index.types import (
    DomainSpec,
    IndexMap,
    OperandDomains,
    Preimage,
    domains,
    flatten_domain_graph_arrays,
    unflatten_domain_graph_arrays,
)

__all__ = [
    "DomainGraph",
    "DomainSpec",
    "IndexMap",
    "OperandDomains",
    "Preimage",
    "domains",
    "flatten_domain_graph_arrays",
    "make_index_map_dag",
    "make_preimages",
    "unflatten_domain_graph_arrays",
]
