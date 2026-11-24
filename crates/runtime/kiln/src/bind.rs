// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::c_void;

use cuda_driver::{CUdeviceptr, CUstream, DeviceBuffer, StreamRef};

use recipe::algebra::Launches;
use recipe::{Address, Arg, BufRef, RecipeBody, Segment};

use crate::Error;

pub(crate) struct Ports<'a> {
    pub(crate) inputs: &'a [CUdeviceptr],
    pub(crate) outputs: &'a [CUdeviceptr],
    pub(crate) workspace: &'a [CUdeviceptr],
}

/// Copy each workspace buffer's declared initial bytes into place.
///
/// The bytes come from `seeds`, the device-resident copy a residency uploaded
/// once when it loaded, rather than from the blob in the recipe. A host
/// pointer would be the shorter route, but a host-to-device copy is not
/// something every caller can replay: recorded into a graph it becomes a node
/// that dereferences that host address again at every launch, and a recipe's
/// blob is not something the launch can be promised still owns. A
/// device-to-device copy carries no host address to go stale, so the same call
/// means the same thing whether it executes now or is recorded for later.
pub(crate) fn seed_workspace<F: Launches>(
    body: &RecipeBody<F>,
    seeds: &[Option<DeviceBuffer<u8>>],
    ports: &Ports,
    stream: CUstream,
) -> Result<(), Error> {
    let stream = StreamRef::from_raw(stream);
    for (i, buf) in body.workspace().iter().enumerate() {
        if buf.init.is_none() {
            continue;
        }
        let seed = seeds
            .get(i)
            .and_then(Option::as_ref)
            .ok_or(Error::WorkspaceUnbound(i))?;
        let dst = workspace_ptr(ports, i)?;
        stream
            .memcpy_d2d(dst, seed.ptr(), seed.nbytes())
            .map_err(Error::Driver)?;
    }
    Ok(())
}

pub(crate) struct KernelParams {
    _values: Vec<u8>,
    params: Vec<*mut c_void>,
}

impl KernelParams {
    pub(crate) fn as_slice(&self) -> &[*mut c_void] {
        &self.params
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [*mut c_void] {
        &mut self.params
    }
}

pub(crate) fn kernel_params(args: &[Arg], ports: &Ports) -> Result<KernelParams, Error> {
    const ALIGN: usize = 8;
    let capacity = args
        .iter()
        .map(|arg| {
            arg.segments()
                .iter()
                .map(|segment| match segment {
                    Segment::Bytes(bytes) => bytes.len(),
                    Segment::Pointer(_) => size_of::<CUdeviceptr>(),
                })
                .sum::<usize>()
                + ALIGN
                - 1
        })
        .sum();
    let mut values = Vec::with_capacity(capacity);
    let base = values.as_mut_ptr();
    let mut params = Vec::with_capacity(args.len());
    for arg in args {
        values.resize(values.len().next_multiple_of(ALIGN), 0);
        params.push(unsafe { base.add(values.len()) }.cast());
        for segment in arg.segments() {
            match segment {
                Segment::Bytes(bytes) => values.extend_from_slice(bytes),
                Segment::Pointer(address) => {
                    let ptr = range_start(ports, *address, 0)?;
                    values.extend_from_slice(&ptr.to_le_bytes());
                }
            }
        }
    }
    debug_assert!(values.len() <= capacity);

    Ok(KernelParams {
        _values: values,
        params,
    })
}

pub(crate) fn bufref_ptr(ports: &Ports, buf: BufRef) -> Result<CUdeviceptr, Error> {
    match buf {
        BufRef::Input(i) => input_ptr(ports, i),
        BufRef::Output(i) => output_ptr(ports, i),
        BufRef::Workspace(i) => workspace_ptr(ports, i),
    }
}

/// Where an operation of `nbytes` starting at `address` begins on the device,
/// once both sums it depends on are proved: the recipe's own
/// `byte_offset + nbytes`, and the bound buffer's base plus every byte the
/// operation reaches from there.
///
/// Every memory reference kiln realizes resolves here, so neither sum is added
/// anywhere else. Proving what the operation *reaches* rather than only where
/// it starts is what makes this a range: a start that fits says nothing about
/// a length that does not, and the driver is handed a base and a count.
///
/// The device bound is the last byte touched, not the exclusive end — a range
/// covering the final bytes of the address space is legal even though one past
/// it has no address. The recipe-level sum is the exclusive end, because an
/// extent is what that one is compared against.
///
/// A kernel pointer argument is the `nbytes = 0` case. How far the kernel then
/// reads is the kernel's business — the recipe does not state it — so all that
/// is proved for one is that the address it is handed exists.
pub(crate) fn range_start<B: Copy + Into<BufRef>>(
    ports: &Ports,
    address: Address<B>,
    nbytes: usize,
) -> Result<CUdeviceptr, Error> {
    address.end(nbytes).ok_or(Error::RangeOverflow {
        byte_offset: address.byte_offset,
        nbytes,
    })?;
    let base = bufref_ptr(ports, address.buf.into())?;
    let overflowed = || Error::AddressOverflow {
        base,
        byte_offset: address.byte_offset,
        nbytes,
    };
    let start = base
        .checked_add(address.byte_offset as CUdeviceptr)
        .ok_or_else(overflowed)?;
    start
        .checked_add(nbytes.saturating_sub(1) as CUdeviceptr)
        .ok_or_else(overflowed)?;
    Ok(start)
}

pub(crate) fn input_ptr(ports: &Ports, index: usize) -> Result<CUdeviceptr, Error> {
    ports
        .inputs
        .get(index)
        .copied()
        .ok_or(Error::InputUnbound(index))
}

pub(crate) fn output_ptr(ports: &Ports, index: usize) -> Result<CUdeviceptr, Error> {
    ports
        .outputs
        .get(index)
        .copied()
        .ok_or(Error::OutputUnbound(index))
}

pub(crate) fn workspace_ptr(ports: &Ports, index: usize) -> Result<CUdeviceptr, Error> {
    ports
        .workspace
        .get(index)
        .copied()
        .ok_or(Error::WorkspaceUnbound(index))
}

#[cfg(test)]
mod tests {
    use super::*;
    use recipe::WritableBuf;

    /// One buffer per kind, all bound at `base`. Extents live in the recipe,
    /// never here — these are the bare base pointers a caller hands in.
    fn resolve(base: CUdeviceptr, byte_offset: usize, nbytes: usize) -> Result<CUdeviceptr, Error> {
        let bound = [base];
        let ports = Ports {
            inputs: &bound,
            outputs: &bound,
            workspace: &bound,
        };
        range_start(&ports, WritableBuf::Output(0).at(byte_offset), nbytes)
    }

    fn address_overflow(base: CUdeviceptr, byte_offset: usize, nbytes: usize) -> bool {
        matches!(
            resolve(base, byte_offset, nbytes),
            Err(Error::AddressOverflow { base: b, byte_offset: o, nbytes: n })
                if (b, o, n) == (base, byte_offset, nbytes)
        )
    }

    fn range_overflow(byte_offset: usize, nbytes: usize) -> bool {
        matches!(
            resolve(0x1000, byte_offset, nbytes),
            Err(Error::RangeOverflow { byte_offset: o, nbytes: n })
                if (o, n) == (byte_offset, nbytes)
        )
    }

    /// The start is the base plus the offset, and it is the start the driver is
    /// handed — the length is proved, never added in.
    #[test]
    fn a_range_resolves_to_where_it_starts() {
        assert_eq!(resolve(0x1000, 0, 64).unwrap(), 0x1000);
        assert_eq!(resolve(0x1000, 16, 64).unwrap(), 0x1010);
    }

    /// A zero-byte operation reaches nothing, so its start is all there is to
    /// prove — including at an offset one past everything before it.
    #[test]
    fn a_zero_byte_range_is_its_start() {
        assert_eq!(resolve(0x1000, 0, 0).unwrap(), 0x1000);
        assert_eq!(resolve(0x1000, 64, 0).unwrap(), 0x1040);
        assert_eq!(
            resolve(0, usize::MAX, 0).unwrap(),
            usize::MAX as CUdeviceptr
        );
    }

    /// The exact end of the address space is reachable; one byte past it is
    /// not. The two differ by the last byte the operation touches rather than
    /// by where it starts, which is what proving the end buys.
    #[test]
    fn the_exact_end_of_the_address_space_resolves_and_one_past_does_not() {
        let last_page = u64::MAX - 63;
        assert_eq!(resolve(last_page, 0, 64).unwrap(), last_page);
        assert_eq!(resolve(last_page, 63, 1).unwrap(), u64::MAX);
        assert!(
            address_overflow(last_page, 0, 65),
            "a start that fits says nothing about a length that does not"
        );
        assert!(address_overflow(last_page, 1, 64));
    }

    /// A base and a range that each fit alone but not together.
    #[test]
    fn a_base_plus_an_in_range_end_can_still_leave_the_address_space() {
        assert!(address_overflow(u64::MAX - 15, 0, 32));
    }

    /// `byte_offset + nbytes` is the recipe's own arithmetic, so it fails
    /// before any buffer is looked at and names no base.
    #[test]
    fn a_range_that_cannot_be_added_is_refused_before_the_binding() {
        assert!(range_overflow(8, usize::MAX));
        assert!(range_overflow(usize::MAX, 1));
    }

    /// The two failures say different things about whose fault it is, and a
    /// selection may only be published under a program identity for the one
    /// that is the program's.
    #[test]
    fn the_two_overflows_are_attributed_differently() {
        assert!(
            Error::RangeOverflow {
                byte_offset: 0,
                nbytes: 0
            }
            .is_program_property(),
            "the recipe's own two numbers are the program's"
        );
        assert!(
            !Error::AddressOverflow {
                base: 0,
                byte_offset: 0,
                nbytes: 0
            }
            .is_program_property(),
            "where a caller bound the buffer is this machine's fact"
        );
    }

    /// A pointer argument states no length, so what is proved for it is that
    /// the address exists — the same resolver at `nbytes = 0`. An offset the
    /// kernel then reads past is the kernel's business, not the recipe's.
    #[test]
    fn a_pointer_argument_proves_its_address_and_no_length() {
        let bound = [0x2000];
        let ports = Ports {
            inputs: &bound,
            outputs: &bound,
            workspace: &bound,
        };
        let params = kernel_params(&[Arg::pointer(BufRef::Input(0).at(24))], &ports).unwrap();
        assert_eq!(params._values, 0x2018u64.to_le_bytes().to_vec());
    }

    #[test]
    fn an_unbound_port_is_named_before_any_arithmetic() {
        let ports = Ports {
            inputs: &[],
            outputs: &[],
            workspace: &[],
        };
        assert!(matches!(
            range_start(&ports, BufRef::Input(2).at(0), 0),
            Err(Error::InputUnbound(2))
        ));
    }
}
