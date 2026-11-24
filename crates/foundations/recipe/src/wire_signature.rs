// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::WireError;
use super::wire_io::{Cursor, Writer};
use crate::{PortAliases, RecipeSignature};

pub(super) const SECTION_SIGNATURE: u32 = 4;
const PORT_STRIDE: usize = 8;
const ALIAS_STRIDE: usize = 8;

pub(super) fn write_signature(
    w: &mut Writer,
    signature: &RecipeSignature,
) -> Result<(), WireError> {
    write_ports(w, signature.inputs())?;
    write_ports(w, signature.outputs())?;
    write_aliases(w, signature.aliases())
}

pub(super) fn write_aliases(w: &mut Writer, aliases: &PortAliases) -> Result<(), WireError> {
    w.count(aliases.as_map().len())?;
    for (input, output) in aliases.iter() {
        w.u32(super::checked_u32(input, "alias input")?);
        w.u32(super::checked_u32(output, "alias output")?);
    }
    Ok(())
}

fn write_ports(w: &mut Writer, ports: &[usize]) -> Result<(), WireError> {
    w.count(ports.len())?;
    for &nbytes in ports {
        w.u64(super::encode_nbytes(nbytes, "port byte extent")?);
    }
    Ok(())
}

pub(super) fn read_signature(c: &mut Cursor<'_>) -> Result<RecipeSignature, WireError> {
    let inputs = read_ports(c)?;
    let outputs = read_ports(c)?;
    let aliases = read_aliases(c, inputs.len(), outputs.len())?;
    Ok(RecipeSignature::with_aliases(inputs, outputs, aliases))
}

pub(super) fn read_aliases(
    c: &mut Cursor<'_>,
    inputs: usize,
    outputs: usize,
) -> Result<PortAliases, WireError> {
    let aliases = (0..c.bounded_count(ALIAS_STRIDE)?)
        .map(|_| Ok((c.u32()? as usize, c.u32()? as usize)))
        .collect::<Result<Vec<_>, WireError>>()?;
    PortAliases::new(aliases, inputs, outputs).map_err(|_| WireError::NonCanonical {
        what: "port aliases",
    })
}

fn read_ports(c: &mut Cursor<'_>) -> Result<Vec<usize>, WireError> {
    let n = c.bounded_count(PORT_STRIDE)?;
    let mut ports = Vec::with_capacity(n);
    for _ in 0..n {
        ports.push(super::decode_nbytes(c.u64()?, "port byte extent")?);
    }
    Ok(ports)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refuses(bytes: &[u8]) {
        let mut c = Cursor::new(bytes, 0);
        assert!(matches!(read_signature(&mut c), Err(WireError::Truncated)));
    }

    #[test]
    fn a_port_count_larger_than_the_section_is_truncation() {
        refuses(&u32::MAX.to_le_bytes());
    }

    #[test]
    fn an_output_count_larger_than_the_rest_of_the_section_is_truncation() {
        let mut bytes = 1u32.to_le_bytes().to_vec();
        bytes.extend_from_slice(&64u64.to_le_bytes());
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        refuses(&bytes);
    }

    #[test]
    fn a_count_whose_bytes_are_one_short_is_truncation() {
        let mut bytes = 2u32.to_le_bytes().to_vec();
        bytes.extend_from_slice(&64u64.to_le_bytes());
        bytes.extend_from_slice(&[0; 7]);
        refuses(&bytes);
    }

    #[test]
    fn an_extent_larger_than_an_object_is_refused() {
        let mut bytes = 1u32.to_le_bytes().to_vec();
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        let mut c = Cursor::new(&bytes, 0);
        assert_eq!(
            read_signature(&mut c),
            Err(WireError::Oversize {
                what: "port byte extent",
                value: u64::MAX,
            })
        );
    }

    #[test]
    fn the_ports_and_aliases_survive_their_own_codec() {
        let aliases = PortAliases::new([(0, 1)], 2, 2).unwrap();
        let signature = RecipeSignature::with_aliases(vec![48, 96], vec![16, 48], aliases);
        let mut w = Writer::new();
        write_signature(&mut w, &signature).unwrap();
        let mut c = Cursor::new(&w.bytes, 0);
        assert_eq!(read_signature(&mut c).unwrap(), signature);
        assert_eq!(c.remaining(), 0);
    }

    #[test]
    fn a_portless_recipe_is_three_zero_counts() {
        let mut w = Writer::new();
        write_signature(&mut w, &RecipeSignature::empty()).unwrap();
        assert_eq!(w.bytes, vec![0; 12]);
    }

    #[test]
    fn a_port_larger_than_four_gibibytes_survives() {
        let big = 5usize << 30;
        let signature = RecipeSignature::new(vec![big], Vec::new());
        let mut w = Writer::new();
        write_signature(&mut w, &signature).unwrap();
        let mut c = Cursor::new(&w.bytes, 0);
        assert_eq!(read_signature(&mut c).unwrap().inputs(), [big]);
    }
}
