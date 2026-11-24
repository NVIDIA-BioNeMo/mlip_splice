// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Assembling a recipe whose work is kernel launches.
//!
//! The structural builder takes a call and what it does. A producer of device
//! work states both together: the buffer it passes is the buffer it reads, so
//! `read`, `write` and `read_write` each add one pointer argument *and* the
//! effect that argument is there for. That is what keeps the argument list and
//! the dependency edges from being two statements of one fact.
//!
//! [`Kernel::pointer`] is the one place they come apart, and it says so: a
//! producer that has proven its kernel never touches the memory behind an
//! address passes the address and declares nothing.

use crate::algebra::builder::{Placement, RecipeBuilder};
use crate::algebra::computation::Geometry;
use crate::algebra::data::RecipeFragment;
use crate::algebra::error::RecipeError;
use crate::algebra::executable::{ExecutableRefusal, KernelLaunch, Launches, ModuleTable};
use crate::arg::{Arg, Segment};
use crate::buffer::BufRef;
use crate::code::{CompileRequest, CompileRequestId, KernelModule, ModuleImage, ResourceBounds};
use crate::effects::Access;
use crate::handles::{Buf, ModuleId, NodeId, Writable};

impl<F: Launches> RecipeBuilder<F> {
    /// A module the loader is handed as it stands.
    pub fn module(&mut self, cubin: impl Into<crate::Blob>) -> ModuleId {
        self.module_definition(KernelModule::from_cuda_loadable(cubin))
    }

    pub fn module_definition(&mut self, module: KernelModule) -> ModuleId {
        let modules = self.body_context_mut().modules_mut();
        modules.push(module);
        ModuleId::from_index(modules.len() - 1)
    }

    /// State a compile, reusing the entry that already states it. Candidates
    /// that share a kernel share its request, so the table a recipe carries has
    /// one entry per distinct image however many modules resolve from it.
    pub fn compile_request(&mut self, request: CompileRequest) -> CompileRequestId {
        self.root_context_mut().intern(request)
    }

    /// A module compiled from `request`, with `images` to try before running
    /// the compile. Their order is the priority the runtime honours.
    pub fn requested_module(
        &mut self,
        request: CompileRequestId,
        images: Vec<ModuleImage>,
        resource_bounds: ResourceBounds,
    ) -> ModuleId {
        self.module_definition(KernelModule::from_request(request, images, resource_bounds))
    }

    pub fn kernel(&mut self, module: ModuleId, entry: &str) -> Kernel<'_, F> {
        Kernel {
            builder: self,
            module,
            entry: entry.to_string(),
            geometry: Geometry::single(),
            args: Vec::new(),
            effects: Vec::new(),
            extra_deps: Vec::new(),
        }
    }

    /// A choice over candidates that are each bound to this recipe's own ports,
    /// in order, and answer for all of them — which is what a producer whose
    /// alternatives are whole programs states.
    pub fn whole_recipe_choice(
        &mut self,
        candidates: Vec<RecipeFragment<F>>,
        inputs: usize,
        outputs: usize,
    ) -> Result<NodeId, RecipeError<ExecutableRefusal>> {
        let inputs: Vec<Buf> = (0..inputs).map(Buf::Input).collect();
        let outputs: Vec<Buf> = (0..outputs).map(Buf::Output).collect();
        self.choice(candidates)
            .inputs(inputs.iter().copied())
            .does(inputs, Access::Read)
            .outputs(outputs.iter().copied())
            .does(outputs, Access::Write)
            .add()
    }
}

/// A kernel launch under construction: what to run, how it is launched, what it
/// is handed, and what it does with it.
#[must_use]
pub struct Kernel<'a, F: Launches> {
    builder: &'a mut RecipeBuilder<F>,
    module: ModuleId,
    entry: String,
    geometry: Geometry,
    args: Vec<Arg>,
    effects: Vec<(Buf, Access)>,
    extra_deps: Vec<NodeId>,
}

impl<F: Launches> Kernel<'_, F> {
    pub fn grid(mut self, grid: [u32; 3]) -> Self {
        self.geometry.grid = grid;
        self
    }

    pub fn block(mut self, block: [u32; 3]) -> Self {
        self.geometry.block = block;
        self
    }

    pub fn shmem(mut self, shmem: u32) -> Self {
        self.geometry.dynamic_shmem = shmem;
        self
    }

    pub fn read(self, buf: impl Into<Buf>) -> Self {
        self.read_at(buf, 0)
    }

    pub fn write(self, buf: impl Into<Writable>) -> Self {
        self.write_at(buf, 0)
    }

    pub fn read_write(self, buf: impl Into<Writable>) -> Self {
        self.read_write_at(buf, 0)
    }

    /// Pass a buffer's address without touching it. The kernel receives the
    /// pointer, and no dependency edge forms against the buffer behind it —
    /// which is what a producer that has *proven* its kernel never reaches that
    /// memory is entitled to state.
    pub fn pointer(self, buf: impl Into<Buf>) -> Self {
        self.pointer_at(buf, 0)
    }

    pub fn pointer_at(mut self, buf: impl Into<Buf>, offset: usize) -> Self {
        self.args
            .push(Arg::pointer(BufRef::from(buf.into()).at(offset)));
        self
    }

    pub fn read_at(self, buf: impl Into<Buf>, offset: usize) -> Self {
        self.handed(buf.into(), offset, Access::Read)
    }

    pub fn write_at(self, buf: impl Into<Writable>, offset: usize) -> Self {
        self.handed(Buf::from(buf.into()), offset, Access::Write)
    }

    pub fn read_write_at(self, buf: impl Into<Writable>, offset: usize) -> Self {
        self.handed(Buf::from(buf.into()), offset, Access::ReadWrite)
    }

    fn handed(mut self, buf: Buf, offset: usize, access: Access) -> Self {
        self.args.push(Arg::pointer(BufRef::from(buf).at(offset)));
        self.effects.push((buf, access));
        self
    }

    pub fn scalar(mut self, bytes: Vec<u8>) -> Self {
        self.args.push(Arg::scalar(bytes));
        self
    }

    pub fn scalar_i32(self, v: i32) -> Self {
        self.scalar(v.to_le_bytes().to_vec())
    }

    pub fn scalar_u32(self, v: u32) -> Self {
        self.scalar(v.to_le_bytes().to_vec())
    }

    pub fn scalar_f32(self, v: f32) -> Self {
        self.scalar(v.to_le_bytes().to_vec())
    }

    /// Pass whole arguments a producer assembled itself, declaring the most
    /// each address it finds in them could be for: everything a body may write
    /// it may be written, and an input it may only be read.
    pub fn args(mut self, args: impl IntoIterator<Item = Arg>) -> Self {
        for arg in args {
            for segment in arg.segments() {
                if let Segment::Pointer(address) = segment {
                    let access = match address.buf {
                        BufRef::Input(_) => Access::Read,
                        BufRef::Output(_) | BufRef::Workspace(_) => Access::ReadWrite,
                    };
                    self.effects.push((Buf::from(address.buf), access));
                }
            }
            self.args.push(arg);
        }
        self
    }

    pub fn after(mut self, deps: &[NodeId]) -> Self {
        self.extra_deps.extend_from_slice(deps);
        self
    }

    pub fn add(self) -> NodeId {
        let Kernel {
            builder,
            module,
            entry,
            geometry,
            args,
            effects,
            extra_deps,
        } = self;
        let call = KernelLaunch {
            module: module.index(),
            entry,
            args,
        };
        let mut placement: Placement<'_, F> = builder.call(call, vec![geometry]).after(&extra_deps);
        for (buf, access) in effects {
            placement = match access {
                Access::Read => placement.reads([buf]),
                Access::Write => placement.writes([buf]),
                Access::ReadWrite => placement.read_writes([buf]),
            };
        }
        // Each effect above was stated by the same call that pushed the
        // pointer it is about, and a store can only be stated through
        // `Writable`, so both halves of the proof hold by construction.
        placement
            .add()
            .expect("a launch declares only stores it is handed somewhere to put")
    }
}
