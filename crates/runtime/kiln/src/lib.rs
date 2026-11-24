// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod artifact_cache;
mod bind;
mod binding_check;
mod compile;
mod device;
mod exec;
mod identity;
mod layout;
mod progress;
mod project;
mod realize;
mod resident;
mod resolve;
mod selection;
mod tune;
mod validate;

use cuda_driver::CUdeviceptr;

pub use artifact_cache::{
    ArtifactCache, ArtifactCacheConfig, ArtifactCacheOutcome, ArtifactKey, CachedArtifact,
    DeviceIdentity, Storable, TargetIdentity, WorkloadIdentity,
};
pub use binding_check::{BindingError, Side, check_contract};
pub use compile::{CompileFailure, CompileFailureKind};
pub use exec::{execute, stats};
pub use identity::RecipeIdentity;
pub use progress::{
    CandidateFailure, CandidateOutcome, CandidateReport, ChoiceProgress, ChoiceReport, TuneReport,
    TuningProgress, Under,
};
pub use realize::{
    ArtifactPolicy, RealizationArgs, RealizationPolicy, RealizationReport, RealizationResult,
    RealizationTiming, Realizer, RealizerOptions, UnrealizedArm,
};
pub use resident::ResidentRecipe;
pub use resolve::{RequestOutcome, RequestReport, ResolutionReport};
pub use selection::{Selection, covering_selections, first_selection, inline};
pub use tune::{TuningPolicy, kernel_entries, tune, tune_reported};
pub use validate::{
    OutputComparator, OutputInitializer, StartingState, ValidationContract, ValidationOutput,
    ValidationStatus, ZeroInitialized,
};

pub struct Bindings<'a> {
    pub inputs: &'a [CUdeviceptr],
    pub outputs: &'a [CUdeviceptr],
    pub workspace: CUdeviceptr,
}

#[derive(Clone, Debug)]
pub enum Error {
    Driver(cuda_driver::Error),
    MissingSymbol(String),
    Decode(String),
    InputUnbound(usize),
    OutputUnbound(usize),
    WorkspaceUnbound(usize),
    /// An operation's own start plus its length leaves the address space, so
    /// there is no range for a binding to resolve.
    RangeOverflow {
        byte_offset: usize,
        nbytes: usize,
    },
    /// The buffer bound here plus that range leaves the device address space.
    AddressOverflow {
        base: CUdeviceptr,
        byte_offset: usize,
        nbytes: usize,
    },
    UnresolvedChoice {
        node: usize,
    },
    /// A replay was checked against a residency that states no ports.
    ///
    /// One loaded from a whole recipe states them; one loaded from a body alone
    /// is a candidate the tuner binds itself, and nothing outside hands it a
    /// caller's buffers.
    UnboundResidency,
    UnresolvedModule {
        module: usize,
    },
    ModuleResolution {
        module: usize,
        failures: Vec<CompileFailure>,
    },
    ArtifactPolicy(String),
    InvalidTuningPolicy(&'static str),
    /// A recipe that does not describe a runnable program reached the point
    /// where kiln would compile, tune, load, or launch it.
    InvalidRecipe(recipe::RecipeError),
    ValidationContract(String),
    /// A candidate ran cleanly and produced something the caller does not
    /// accept as the program's result.
    CandidateDisagrees {
        candidate: usize,
        output: usize,
    },
    ExhaustedChoice {
        node: usize,
        failures: Vec<Error>,
    },
    ContextMismatch,
}

impl Error {
    /// Whether this failure belongs to the program rather than to the machine
    /// or the moment. A candidate eliminated by something that is not the
    /// program — an absent toolchain, an interrupted worker, a driver error, a
    /// source that contradicts its own digest — leaves a selection that another
    /// machine would not have made, so that selection must not be published
    /// under the program's identity.
    pub fn is_program_property(&self) -> bool {
        match self {
            Error::ModuleResolution { failures, .. } => failures
                .iter()
                .all(|failure| failure.kind.is_program_property()),
            Error::ExhaustedChoice { failures, .. } => {
                failures.iter().all(Error::is_program_property)
            }
            // Where the caller bound the buffer decides whether its base plus
            // the recipe's range still fits an address, so this is a fact about
            // this binding rather than about the program.
            Error::Driver(_)
            | Error::ContextMismatch
            | Error::AddressOverflow { .. }
            | Error::InvalidTuningPolicy(_) => false,
            Error::MissingSymbol(_)
            | Error::Decode(_)
            | Error::InputUnbound(_)
            | Error::OutputUnbound(_)
            | Error::WorkspaceUnbound(_)
            // The two numbers are the recipe's own, so no other machine adds
            // them to anything else.
            | Error::RangeOverflow { .. }
            | Error::UnresolvedChoice { .. }
            | Error::UnboundResidency
            | Error::UnresolvedModule { .. }
            | Error::ArtifactPolicy(_)
            | Error::ValidationContract(_)
            | Error::InvalidRecipe(_)
            | Error::CandidateDisagrees { .. } => true,
        }
    }
}
