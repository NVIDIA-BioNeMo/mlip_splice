// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::iteration::Plan as DomainPlan;
use super::super::partition::KernelPart;
use super::PartBoundary;
use crate::eval::GuardKey;
use gpu_ir::Scalar;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::backend) enum WriteMode {
    Store,
    AddOrdered,
    AddAtomic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(in crate::backend) enum ReductionProvenance {
    Reduce,
    Dot,
    HiddenLane,
    Fiber,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::backend) struct ReductionRegion {
    pub provenance: ReductionProvenance,
    pub output: usize,
    pub element: usize,
    pub scalar: Scalar,
    pub launches: Vec<usize>,
    pub init_launch: usize,
}

#[derive(Clone, Debug)]
pub(in crate::backend) struct GuardedRegion {
    pub key: GuardKey,
    pub outputs: Vec<(usize, usize)>,
    pub part: KernelPart,
}

#[derive(Clone, Debug)]
pub(in crate::backend) struct GuardedPlan {
    pub regions: Vec<GuardedRegion>,
    pub remainder: KernelPart,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::backend) struct MaterializedValue {
    /// Which binding of the program is cut here.
    ///
    /// A name and not a value: the cut is chosen against the planning profile
    /// and resolved again in each kernel's own evaluation, and values never
    /// cross arenas. A binding the profile does not name cannot be cut, because
    /// there would be nothing to look for on the other side.
    pub name: String,
    pub scalar: Scalar,
    pub nbytes: usize,
    pub closure_statements: usize,
    pub consumer_count: usize,
}

#[derive(Clone, Debug)]
pub(in crate::backend) enum Launch {
    Producer {
        output: usize,
        inputs: Vec<usize>,
    },
    Consumer {
        part: KernelPart,
        output_modes: Vec<WriteMode>,
    },
}

/// Which physics motivated a candidate's cut: structural candidates relieve
/// register/footprint pressure and compete on the seam metric; the traffic
/// candidate relieves L2 residency and earns its compile slot by bytes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::backend) enum CandidateOrigin {
    Structural,
    Traffic,
}

#[derive(Clone, Debug)]
pub(in crate::backend) struct Candidate {
    pub materialized: Vec<MaterializedValue>,
    pub launches: Vec<Launch>,
    pub reductions: Vec<ReductionRegion>,
    pub guards: Vec<Option<GuardedPlan>>,
    pub hoists: Vec<bool>,
    pub origin: CandidateOrigin,
}

pub(in crate::backend) struct PlanIr {
    pub domain: DomainPlan,
    pub candidates: Vec<Candidate>,
    /// The partition this plan's baseline candidate launches, one entry per
    /// consumer part. Recorded only when the compile asked for the structural
    /// baseline; empty otherwise.
    pub boundaries: Vec<PartBoundary>,
}
