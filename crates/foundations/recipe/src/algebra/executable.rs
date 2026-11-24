// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The computation family the CUDA runtime executes.
//!
//! A recipe's root data is the compile-request graph: one entry per distinct
//! compile, with the prerequisites each one presupposes. A body's data is its
//! module table, and each module either names a request or carries its own
//! images. A call is a kernel launch: a module, an entry point, the geometry it
//! runs under, and the arguments it is handed.
//!
//! That layering is what the structural core relies on and never has to know:
//! requests are named by modules, and a launch names only a module of the body
//! it sits in. So merging two recipes rebases request handles inside module
//! tables and module handles inside launches, and nothing else moves.

use std::collections::HashMap;
use std::fmt;

use crate::algebra::computation::{
    Addresses, BytePortContract, Computation, Geometry, Identify, IdentityWalk,
};
use crate::arg::{Arg, Segment};
use crate::buffer::{Address, BufRef};
use crate::code::{CompileRequest, CompileRequestId, KernelCode, KernelModule};
use crate::requests::{RequestKey, dependency_order, request_key, same_unit, shift_requires};
use crate::signature::RecipeSignature;

/// The family of recipes that run on a device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Executable;

/// Where a body of launches keeps the modules its launches name.
///
/// A launch names a module by position in the table of the body that holds it,
/// so the runtime needs that table and nothing else. What else a family keeps
/// beside it — what a candidate is called, say — is the family's, and stating
/// where the modules are is what keeps the two from being one thing.
///
/// The table is reached mutably as well, because building a body appends to it
/// and merging one rebases what it holds. Nothing outside replaces it whole:
/// what a body context is beyond its modules is never the runtime's to invent.
pub trait ModuleTable: Clone + fmt::Debug + Default + PartialEq + Eq {
    fn modules(&self) -> &[KernelModule];
    fn modules_mut(&mut self) -> &mut Vec<KernelModule>;
}

impl ModuleTable for Vec<KernelModule> {
    fn modules(&self) -> &[KernelModule] {
        self
    }

    fn modules_mut(&mut self) -> &mut Vec<KernelModule> {
        self
    }
}

/// A family whose work is kernel launches: a compile-request graph for root
/// data, a module table for each body, and a launch for each call.
///
/// What such a family is still free to choose is the contract it is bound
/// through and whatever it keeps beside its module table — a runtime needs only
/// the byte extents of the ports and where the modules are, and everything
/// above it may state far more. So this is the bound the runtime is written
/// against: it fixes what a call *is* without fixing what the recipe holding it
/// promises its caller or what it calls the bodies below it.
pub trait Launches:
    Computation<
        RootContext = CompileRequests,
        BodyContext: ModuleTable,
        Call = KernelLaunch,
        RootRemap = RequestRemap,
        BodyRemap = ModuleBase,
        Refusal = ExecutableRefusal,
    >
{
}

impl Launches for Executable {}

/// One kernel launch: what to run, where it comes from, and what it is handed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelLaunch {
    /// Position in the module table of the body this launch sits in.
    pub module: usize,
    pub entry: String,
    pub args: Vec<Arg>,
}

/// A recipe's compile table: one entry per distinct compile, and the index that
/// keeps it that way.
///
/// The index is derived from the entries — it says which of them are worth
/// comparing — so it is not part of the value: two tables holding the same
/// requests are the same table however each was assembled.
#[derive(Clone, Default)]
pub struct CompileRequests {
    requests: Vec<CompileRequest>,
    /// The entries each key buckets, in the order they were stated: one key can
    /// hold several requests, since equal claims are not equal bytes.
    by_key: HashMap<RequestKey, Vec<usize>>,
}

/// Where an incoming recipe's compile requests landed in the merged table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestRemap(Vec<usize>);

/// Where an incoming body's module table starts in the merged one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModuleBase(pub usize);

/// What this family refuses about the data it owns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutableRefusal {
    /// A compile that links nothing. An assembler has no answer for it, so it
    /// is refused where the recipe is proved rather than where one is run.
    EmptyCompileUnit {
        request: usize,
    },
    RequestOutOfRange {
        request: usize,
        referenced: usize,
    },
    CompileRequestCycle {
        request: usize,
    },
    ModuleRequestOutOfRange {
        module: usize,
        referenced: usize,
    },
    ModuleOutOfRange {
        module: usize,
    },
}

impl fmt::Display for ExecutableRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExecutableRefusal::EmptyCompileUnit { request } => {
                write!(f, "compile request {request} states no source to assemble")
            }
            ExecutableRefusal::RequestOutOfRange {
                request,
                referenced,
            } => write!(
                f,
                "compile request {request} references compile request {referenced}, out of range"
            ),
            ExecutableRefusal::CompileRequestCycle { request } => write!(
                f,
                "compile request dependency cycle reaches request {request}"
            ),
            ExecutableRefusal::ModuleRequestOutOfRange { module, referenced } => write!(
                f,
                "module {module} references compile request {referenced}, out of range"
            ),
            ExecutableRefusal::ModuleOutOfRange { module } => {
                write!(f, "references module {module}, out of range")
            }
        }
    }
}

impl std::error::Error for ExecutableRefusal {}

impl BytePortContract for RecipeSignature {
    fn num_inputs(&self) -> usize {
        RecipeSignature::num_inputs(self)
    }

    fn num_outputs(&self) -> usize {
        RecipeSignature::num_outputs(self)
    }

    fn input_nbytes(&self, port: usize) -> Option<usize> {
        RecipeSignature::input_nbytes(self, port)
    }

    fn output_nbytes(&self, port: usize) -> Option<usize> {
        RecipeSignature::output_nbytes(self, port)
    }

    fn aliased_output(&self, input: usize) -> Option<usize> {
        self.aliases().output_for_input(input)
    }
}

/// The port extents, framed as [`crate::RecipeSignature`] states them: inputs
/// then outputs, each run counted before its widths.
impl Identify for RecipeSignature {
    fn identify(&self, walk: &mut dyn IdentityWalk) {
        walk.count(self.num_inputs());
        for &nbytes in self.inputs() {
            walk.number(nbytes as u64);
        }
        walk.count(self.num_outputs());
        for &nbytes in self.outputs() {
            walk.number(nbytes as u64);
        }
        walk.count(self.aliases().as_map().len());
        for (input, output) in self.aliases().iter() {
            walk.count(input);
            walk.count(output);
        }
    }
}

/// A launch is handed pointers, and a kernel forms its own addresses from
/// there. What it does through them is the placement's declared effects, not
/// something read off the argument list, so this states only where it reaches.
impl Addresses for KernelLaunch {
    fn addresses(&self, visit: &mut dyn FnMut(Address<BufRef>)) {
        for arg in &self.args {
            for segment in arg.segments() {
                if let Segment::Pointer(address) = segment {
                    visit(*address);
                }
            }
        }
    }

    fn rebind(&mut self, bind: &mut dyn FnMut(Address<BufRef>) -> Address<BufRef>) {
        // Replacing a pointer never joins or empties a byte run, so the result
        // is canonical whenever the argument was — which the type guarantees.
        for arg in &mut self.args {
            *arg = Arg::from_segments(arg.segments().iter().map(|segment| match segment {
                Segment::Bytes(bytes) => Segment::Bytes(bytes.clone()),
                Segment::Pointer(address) => Segment::Pointer(bind(*address)),
            }));
        }
    }
}

impl RequestRemap {
    /// Where the entry that was `request` now sits. One the remap does not
    /// cover is sent out of range, so validation reports it rather than letting
    /// it name whichever entry happens to sit at that offset.
    pub fn of(&self, request: usize) -> usize {
        self.0.get(request).copied().unwrap_or(usize::MAX)
    }

    pub fn as_slice(&self) -> &[usize] {
        &self.0
    }
}

impl CompileRequests {
    /// The table `requests` are, at exactly the positions they hold. Nothing is
    /// merged: a decoded table's handles name its own offsets.
    pub fn new(requests: Vec<CompileRequest>) -> Self {
        let mut by_key: HashMap<RequestKey, Vec<usize>> = HashMap::new();
        for (index, request) in requests.iter().enumerate() {
            if let Some(key) = request_key(request) {
                by_key.entry(key).or_default().push(index);
            }
        }
        CompileRequests { requests, by_key }
    }

    pub fn as_slice(&self) -> &[CompileRequest] {
        &self.requests
    }

    /// The entries, mutably, for the one pass that rewrites a payload without
    /// changing what any request is: re-owning source bytes leaves every
    /// request's sources, options and prerequisites — and so the index over
    /// them — exactly as they were.
    pub(crate) fn as_mut_slice(&mut self) -> &mut [CompileRequest] {
        &mut self.requests
    }

    pub fn into_vec(self) -> Vec<CompileRequest> {
        self.requests
    }

    pub fn len(&self) -> usize {
        self.requests.len()
    }

    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    pub fn get(&self, request: CompileRequestId) -> Option<&CompileRequest> {
        self.requests.get(request.0)
    }

    /// State a compile, reusing the entry that already states it.
    ///
    /// A key only buckets the entries worth comparing; what merges two is their
    /// sources holding the same bytes. Reusing an entry on a claimed digest
    /// would drop a source no one had read in favour of one that happened to
    /// name itself the same way.
    pub fn intern(&mut self, request: CompileRequest) -> CompileRequestId {
        let key = request_key(&request);
        let existing = key.as_ref().and_then(|key| {
            let bucket = self.by_key.get(key)?;
            bucket
                .iter()
                .copied()
                .find(|&existing| same_unit(&self.requests[existing], &request))
        });
        if let Some(existing) = existing {
            return CompileRequestId(existing);
        }
        self.requests.push(request);
        let id = self.requests.len() - 1;
        if let Some(key) = key {
            self.by_key.entry(key).or_default().push(id);
        }
        CompileRequestId(id)
    }

    /// State a compile as its own entry, whatever the table already holds.
    ///
    /// The one caller is a table that cannot be walked in dependency order: its
    /// entries move across whole, unmerged, so validation reports what is wrong
    /// with them instead of interning renumbering a cycle into something else.
    fn push(&mut self, request: CompileRequest) -> CompileRequestId {
        self.requests.push(request);
        CompileRequestId(self.requests.len() - 1)
    }
}

impl PartialEq for CompileRequests {
    fn eq(&self, other: &Self) -> bool {
        self.requests == other.requests
    }
}

impl Eq for CompileRequests {}

impl fmt::Debug for CompileRequests {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(&self.requests).finish()
    }
}

impl Computation for Executable {
    /// Anyone may make a kernel launch and the extents it is bound through:
    /// this is the family the public format states, and its values are what
    /// its bytes are.
    type Authorization = ();

    type Contract = RecipeSignature;
    type RootContext = CompileRequests;
    type BodyContext = Vec<KernelModule>;
    type Call = KernelLaunch;
    type RootRemap = RequestRemap;
    type BodyRemap = ModuleBase;
    type Refusal = ExecutableRefusal;

    /// Interning in dependency order, so that a prerequisite is already on its
    /// merged index when the request naming it arrives.
    fn merge_root_context(
        into: &mut CompileRequests,
        incoming: CompileRequests,
    ) -> Self::RootRemap {
        let incoming = incoming.into_vec();
        let Some(order) = dependency_order(&incoming) else {
            let base = into.len();
            let mut remap = Vec::with_capacity(incoming.len());
            for mut request in incoming {
                request.requires = shift_requires(&request.requires, |id| id + base);
                remap.push(into.push(request).0);
            }
            return RequestRemap(remap);
        };
        let mut remap = vec![usize::MAX; incoming.len()];
        let mut incoming: Vec<Option<CompileRequest>> = incoming.into_iter().map(Some).collect();
        for index in order {
            let mut request = incoming[index]
                .take()
                .expect("dependency order names each request once");
            request.requires = shift_requires(&request.requires, |id| remap[id]);
            remap[index] = into.intern(request).0;
        }
        RequestRemap(remap)
    }

    fn merge_body_context(
        into: &mut Vec<KernelModule>,
        incoming: Vec<KernelModule>,
    ) -> Self::BodyRemap {
        let base = into.len();
        into.extend(incoming);
        ModuleBase(base)
    }

    fn rebase_body_context(context: &mut Vec<KernelModule>, remap: &RequestRemap) {
        for module in context {
            if let KernelCode::PtxBacked {
                compile_request, ..
            } = &mut module.code
            {
                compile_request.0 = remap.of(compile_request.0);
            }
        }
    }

    fn rebase_call(call: &mut KernelLaunch, remap: &ModuleBase) {
        call.module += remap.0;
    }

    fn validate_root_context(context: &CompileRequests) -> Result<(), ExecutableRefusal> {
        let requests = context.as_slice();
        for (request, compile) in requests.iter().enumerate() {
            if compile.sources.is_empty() {
                return Err(ExecutableRefusal::EmptyCompileUnit { request });
            }
            for prerequisite in compile.requires.keys() {
                if prerequisite.0 >= requests.len() {
                    return Err(ExecutableRefusal::RequestOutOfRange {
                        request,
                        referenced: prerequisite.0,
                    });
                }
            }
        }
        let mut state = vec![Mark::Unvisited; requests.len()];
        for request in 0..requests.len() {
            acyclic(request, requests, &mut state)?;
        }
        Ok(())
    }

    /// A module always has somewhere to get its kernels from — [`KernelCode`]
    /// admits no other shape — so all that is left to prove is that the compile
    /// it names is one this recipe holds.
    fn validate_body_context(
        context: &Vec<KernelModule>,
        root: &CompileRequests,
    ) -> Result<(), ExecutableRefusal> {
        for (module, implementation) in context.iter().enumerate() {
            if let Some(request) = implementation.code.compile_request()
                && request.0 >= root.len()
            {
                return Err(ExecutableRefusal::ModuleRequestOutOfRange {
                    module,
                    referenced: request.0,
                });
            }
        }
        Ok(())
    }

    fn validate_call(
        call: &KernelLaunch,
        context: &Vec<KernelModule>,
    ) -> Result<(), ExecutableRefusal> {
        if call.module >= context.len() {
            return Err(ExecutableRefusal::ModuleOutOfRange {
                module: call.module,
            });
        }
        Ok(())
    }

    /// A kernel launch is one launch. That is what makes it a launch rather
    /// than a composite of them.
    fn launch_arity(_call: &KernelLaunch) -> usize {
        1
    }

    /// Whether an architecture runs the grid and block a launch asks for is
    /// decided where the module is loaded, against the device that will run it.
    /// A recipe holds no device, so there is nothing here this family can add
    /// to what the core already proved.
    fn validate_geometry(
        _call: &KernelLaunch,
        _launch: usize,
        _geometry: &Geometry,
    ) -> Result<(), ExecutableRefusal> {
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Mark {
    Unvisited,
    Visiting,
    Ordered,
}

fn acyclic(
    request: usize,
    requests: &[CompileRequest],
    state: &mut [Mark],
) -> Result<(), ExecutableRefusal> {
    match state[request] {
        Mark::Ordered => return Ok(()),
        Mark::Visiting => return Err(ExecutableRefusal::CompileRequestCycle { request }),
        Mark::Unvisited => {}
    }
    state[request] = Mark::Visiting;
    for prerequisite in requests[request].requires.keys() {
        acyclic(prerequisite.0, requests, state)?;
    }
    state[request] = Mark::Ordered;
    Ok(())
}

#[cfg(test)]
#[path = "executable_tests.rs"]
mod tests;
