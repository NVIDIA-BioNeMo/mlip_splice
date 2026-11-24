// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Content identity of a pending recipe.

use recipe::algebra::{Identify, IdentityWalk, Launches, ModuleTable};
use recipe::{
    Address, Arg, Blob, BlobDecodeError, BufRef, CompileRequest, Geometry, KernelCode,
    KernelLaunch, KernelModule, ModuleImage, Node, PtxCompileOptions, Recipe, RecipeBody,
    ResourceBounds, Segment, Work, WorkspaceBuf, WritableBuf,
};

/// A BLAKE3 digest of everything in a pending recipe that determines the
/// program: the external ports it is bound through, its compile requests, its
/// module images, its workspace, and the node graph of every `Choice`
/// candidate.
///
/// The signature leads because it is what the program is bound through: two
/// recipes that launch identical kernels but state different port extents are
/// different programs, and a cache keyed on this must not hand one to a caller
/// holding buffers sized for the other.
///
/// Structure and payloads are hashed in two passes because they are enumerated
/// in two ways. The structure is walked here; the payloads come from
/// [`recipe::payloads::recipe_payloads`], the one place that says which payloads a
/// recipe carries — the same walk the artifact's blob pool and the detaching
/// pass consume. An identity that enumerated payloads itself would be a second
/// answer to that question, free to miss a field the others handle.
///
/// Payload bytes enter only through [`Blob::content_digest`], which hashes what
/// a payload *holds* rather than what its artifact claims. That distinction is
/// the whole of this digest's worth: an encoded payload's stated digest is
/// unchecked until the bytes are expanded, so an identity built on it would let
/// a recipe naming another program's digest be handed that program's realized
/// artifact — its compiled code — without a byte of its own ever being read.
/// A payload that does not hash to what it states is not identifiable at all,
/// and a caller that cannot identify a recipe neither reads nor writes a cache
/// under it.
///
/// The price is one streaming pass over the recipe's payloads, expanding each
/// compressed source and releasing it. Identifying a gigabyte-scale pending
/// recipe still never serializes it, never retains an expansion and never
/// copies an image, so the cost is time over the payload bytes and no resident
/// memory. A digest that only narrows candidates does not pay it —
/// [`Blob::claimed_digest`] is that digest, and the compile-request bucket a
/// compile is looked up in is where it belongs.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct RecipeIdentity([u8; 32]);

impl RecipeIdentity {
    pub fn of<F: Launches>(recipe: &Recipe<F>) -> Result<Self, BlobDecodeError> {
        Self::over(recipe, recipe::payloads::recipe_payloads(recipe))
    }

    /// The same, over a stated enumeration of the recipe's payloads.
    ///
    /// [`RecipeIdentity::of`] supplies the one walk. This exists so a test can
    /// supply a walk that misses a role and show what an identity built on such
    /// a walk would then fail to tell apart — the failure is silent by nature,
    /// so it has to be produced deliberately to be pinned.
    fn over<'a, F: Launches>(
        recipe: &Recipe<F>,
        payloads: impl Iterator<Item = recipe::payloads::Payload<'a>>,
    ) -> Result<Self, BlobDecodeError> {
        let mut digest = Framed::new("kiln-program-identity-v4");
        recipe.contract().identify(&mut digest);
        digest.count(recipe.context().as_slice().len());
        for request in recipe.context().as_slice() {
            hash_request(&mut digest, request);
        }
        hash_body(&mut digest, recipe.body());
        hash_payloads(&mut digest, payloads)?;
        Ok(RecipeIdentity(digest.finish()))
    }

    /// Scope a loaded artifact to the policy that selects its measured winner.
    /// Unbounded tuning keeps the program identity; equal durations share a key.
    pub fn with_tuning_policy(self, policy: &crate::TuningPolicy) -> Result<Self, &'static str> {
        let Some(limit) = policy.scheduling_time_limit()? else {
            return Ok(self);
        };
        let mut digest = Framed::new("kiln-tuning-policy-v1");
        digest.digest(self.as_bytes());
        digest.number(limit.as_secs());
        digest.number(u64::from(limit.subsec_nanos()));
        Ok(Self(digest.finish()))
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// The identity a caller read back from wherever [`RecipeIdentity::as_bytes`]
    /// wrote one.
    ///
    /// A digest travels: it is written into a compiler artifact and read by a
    /// later process, or stored beside a cached realization. Every such round
    /// trip needs this direction, and it is total on 32 bytes because those
    /// bytes name a program by content rather than by anything this process
    /// could check them against — a digest that names nothing loaded here
    /// resolves to nothing, which is a lookup's answer to give and not a
    /// conversion's.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        RecipeIdentity(bytes)
    }
}

impl std::fmt::Debug for RecipeIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in &self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// A BLAKE3 hasher whose every field is length- or width-framed, so no two
/// different structures can agree by re-cutting one field into the next.
pub(crate) struct Framed(blake3::Hasher);

impl Framed {
    pub(crate) fn new(domain: &str) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&(domain.len() as u64).to_le_bytes());
        hasher.update(domain.as_bytes());
        Framed(hasher)
    }

    pub(crate) fn byte(&mut self, value: u8) -> &mut Self {
        self.0.update(&[value]);
        self
    }

    pub(crate) fn count(&mut self, value: usize) -> &mut Self {
        self.0.update(&(value as u64).to_le_bytes());
        self
    }

    pub(crate) fn number(&mut self, value: u64) -> &mut Self {
        self.0.update(&value.to_le_bytes());
        self
    }

    pub(crate) fn maybe_number(&mut self, value: Option<u64>) -> &mut Self {
        match value {
            Some(value) => self.byte(1).number(value),
            None => self.byte(0),
        }
    }

    pub(crate) fn bytes(&mut self, value: &[u8]) -> &mut Self {
        self.count(value.len());
        self.0.update(value);
        self
    }

    pub(crate) fn digest(&mut self, value: &[u8; 32]) -> &mut Self {
        self.0.update(value);
        self
    }

    pub(crate) fn finish(&self) -> [u8; 32] {
        *self.0.finalize().as_bytes()
    }
}

/// A contract states what makes it different from another, field by field, and
/// this is where those fields land. The framing is the same one every other
/// field of the identity is written with, so what a family says about its
/// contract is hashed the way everything around it is.
impl IdentityWalk for Framed {
    fn byte(&mut self, value: u8) {
        Framed::byte(self, value);
    }

    fn count(&mut self, value: usize) {
        Framed::count(self, value);
    }

    fn number(&mut self, value: u64) {
        Framed::number(self, value);
    }

    fn bytes(&mut self, value: &[u8]) {
        Framed::bytes(self, value);
    }
}

/// Every payload the recipe carries, each framed by what it is.
///
/// The structure pass above hashes how many images each module has and how many
/// candidates each `Choice` holds, which is what fixes where in this sequence a
/// given payload falls — so two recipes whose payloads differ anywhere differ
/// here, and swapping two modules' images is a different program.
///
/// The role is hashed with the bytes because the same bytes serving two
/// purposes are two programs, and because a role that stopped being reached by
/// the walk would then change this digest rather than silently drop out of it.
fn hash_payloads<'a>(
    digest: &mut Framed,
    payloads: impl Iterator<Item = recipe::payloads::Payload<'a>>,
) -> Result<(), BlobDecodeError> {
    let payloads: Vec<_> = payloads.collect();
    digest.count(payloads.len());
    for payload in payloads {
        digest.byte(payload.role.tag());
        hash_blob(digest, payload.blob)?;
    }
    Ok(())
}

fn hash_blob(digest: &mut Framed, blob: &Blob) -> Result<(), BlobDecodeError> {
    digest.count(blob.len()).digest(&blob.content_digest()?);
    Ok(())
}

fn hash_options(digest: &mut Framed, options: &PtxCompileOptions) {
    digest.maybe_number(options.max_registers.map(u64::from));
    digest.byte(match options.assembly_mode {
        recipe::PtxAssemblyMode::EnvironmentDefault => 0,
        recipe::PtxAssemblyMode::Relocatable => 1,
        recipe::PtxAssemblyMode::WholeProgram => 2,
    });
}

fn hash_bounds(digest: &mut Framed, bounds: &ResourceBounds) {
    digest.maybe_number(bounds.min_registers.map(u64::from));
    digest.maybe_number(bounds.max_local_bytes.map(u64::from));
}

/// What a compile asks for, minus the texts it asks it of — those are
/// payloads, and payloads are hashed together at the end.
///
/// The unit is hashed as the sequence it is: two compiles over the same
/// modules in a different link order are two programs, and the count fixes
/// where each source's bytes fall in the payload pass.
fn hash_request(digest: &mut Framed, request: &CompileRequest) {
    digest.count(request.sources.len());
    for source in &request.sources {
        match source.kind {
            recipe::SourceKind::Ptx => digest.byte(0),
        };
        digest
            .byte(source.target.isa[0])
            .byte(source.target.isa[1])
            .number(u64::from(source.target.minimum_sm));
    }
    hash_options(digest, &request.options);
    digest.count(request.requires.len());
    for (prerequisite, bounds) in &request.requires {
        digest.count(prerequisite.0);
        hash_bounds(digest, bounds);
    }
}

/// Everything an image states about itself except its bytes: what it is and
/// what it is for. The same bytes on a different architecture are a different
/// program, and that is said here; the bytes themselves are a payload.
fn hash_image(digest: &mut Framed, image: &ModuleImage) {
    match image {
        ModuleImage::Cubin { target_sm, .. } => {
            digest.byte(0).number(u64::from(*target_sm));
        }
        ModuleImage::CudaLoadable { persistence, .. } => {
            digest.byte(1).byte(match persistence {
                recipe::ImagePersistence::Persistable => 0,
                recipe::ImagePersistence::SessionOnly => 1,
            });
        }
    }
}

/// Where a module's kernels come from, in the order they would be tried.
/// Reordering two images changes which one this machine loads, so the order is
/// hashed rather than the set.
fn hash_module(digest: &mut Framed, module: &KernelModule) {
    match &module.code {
        KernelCode::PtxBacked {
            compile_request,
            images,
        } => {
            digest.byte(0).count(compile_request.0).count(images.len());
            for image in images {
                hash_image(digest, image);
            }
        }
        KernelCode::ImageOnly {
            preferred_image,
            alternative_images,
        } => {
            digest.byte(1);
            hash_image(digest, preferred_image);
            digest.count(alternative_images.len());
            for image in alternative_images {
                hash_image(digest, image);
            }
        }
    }
    hash_bounds(digest, &module.resource_bounds);
}

/// A workspace buffer's extent, and whether it starts life holding something.
/// Whether, not what: the seed bytes are a payload, but a buffer that is seeded
/// and one that is not are different programs even before the bytes are read.
fn hash_workspace(digest: &mut Framed, buffer: &WorkspaceBuf) {
    digest.count(buffer.nbytes);
    digest.byte(u8::from(buffer.init.is_some()));
}

fn hash_buf(digest: &mut Framed, buf: BufRef) {
    let tag = match buf {
        BufRef::Input(_) => 0,
        BufRef::Output(_) => 1,
        BufRef::Workspace(_) => 2,
    };
    digest.byte(tag).count(buf.index());
}

fn hash_writable(digest: &mut Framed, buf: WritableBuf) {
    let tag = match buf {
        WritableBuf::Output(_) => 0,
        WritableBuf::Workspace(_) => 1,
    };
    digest.byte(tag).count(buf.index());
}

/// Where an operation starts is as much of what it does as which buffer it
/// names, so both are in the identity.
fn hash_address(digest: &mut Framed, address: Address<BufRef>) {
    hash_buf(digest, address.buf);
    digest.count(address.byte_offset);
}

fn hash_writable_address(digest: &mut Framed, address: Address<WritableBuf>) {
    hash_writable(digest, address.buf);
    digest.count(address.byte_offset);
}

fn hash_arg(digest: &mut Framed, arg: &Arg) {
    digest.count(arg.segments().len());
    for segment in arg.segments() {
        match segment {
            Segment::Bytes(bytes) => {
                digest.byte(0).bytes(bytes);
            }
            Segment::Pointer(address) => {
                digest.byte(1);
                hash_address(digest, *address);
            }
        }
    }
}

fn hash_node<F: Launches>(digest: &mut Framed, node: &Node<F>) {
    digest.count(node.deps().len());
    for dep in node.deps() {
        digest.count(*dep);
    }
    match node.work() {
        Work::Call {
            call:
                KernelLaunch {
                    module,
                    entry,
                    args,
                },
            geometry,
        } => {
            digest.byte(0).count(*module).bytes(entry.as_bytes());
            // One shape, because this family's calls are one launch. Framing a
            // count around it would say a second thing about a number the
            // family already fixes, and would not be the digest recipes have
            // always had.
            for Geometry {
                grid,
                block,
                dynamic_shmem: shmem,
            } in geometry
            {
                for extent in grid.iter().chain(block) {
                    digest.number(u64::from(*extent));
                }
                digest.number(u64::from(*shmem));
            }
            digest.count(args.len());
            for arg in args {
                hash_arg(digest, arg);
            }
        }
        Work::Memset {
            destination,
            value,
            nbytes,
        } => {
            digest.byte(1);
            hash_writable_address(digest, *destination);
            digest.byte(*value).count(*nbytes);
        }
        Work::Memcpy {
            source,
            destination,
            nbytes,
        } => {
            digest.byte(2);
            hash_address(digest, *source);
            hash_writable_address(digest, *destination);
            digest.count(*nbytes);
        }
        Work::Choice {
            candidates,
            input_binding,
            output_binding,
        } => {
            digest.byte(3).count(input_binding.len());
            for buf in input_binding {
                hash_buf(digest, *buf);
            }
            digest.count(output_binding.len());
            for buf in output_binding {
                hash_buf(digest, *buf);
            }
            digest.count(candidates.len());
            for candidate in candidates {
                hash_body(digest, candidate);
            }
        }
    }
}

fn hash_body<F: Launches>(digest: &mut Framed, body: &RecipeBody<F>) {
    digest.count(body.context().modules().len());
    for module in body.context().modules() {
        hash_module(digest, module);
    }
    digest.count(body.workspace().len());
    for buffer in body.workspace() {
        hash_workspace(digest, buffer);
    }
    digest.count(body.nodes().len());
    for node in body.nodes() {
        hash_node(digest, node);
    }
}

#[cfg(test)]
#[path = "identity_tests.rs"]
mod tests;

#[cfg(test)]
mod tuning_policy_tests {
    use super::RecipeIdentity;
    use crate::TuningPolicy;

    #[test]
    fn tuning_policy_scopes_loaded_artifacts() {
        let program = RecipeIdentity::from_bytes([7; 32]);
        let key = |limit| {
            program.with_tuning_policy(&TuningPolicy {
                scheduling_time_limit_ms: limit,
            })
        };
        assert_eq!(key(None).unwrap(), program);
        assert_ne!(key(Some(0.0)).unwrap(), program);
        assert_ne!(key(Some(0.0)).unwrap(), key(Some(1.0)).unwrap());
        assert_eq!(key(Some(0.0)).unwrap(), key(Some(-0.0)).unwrap());
        assert_eq!(key(Some(1.0)).unwrap(), key(Some(1.0)).unwrap());
        assert!(key(Some(-1.0)).is_err());
        assert!(key(Some(f64::INFINITY)).is_err());
    }
}
