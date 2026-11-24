// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wire format: a content-deduplicated blob pool plus a recursive body.
//!
//! All payload bytes (compile sources, module images, workspace inits) live
//! once in the BLOBS section, deduplicated by content across the whole Choice
//! tree. The BODY section holds the structure — modules and inits reference
//! pool entries by index, Choice candidates nest recursively. Decoding is
//! zero-copy: every payload becomes a [`Blob`] view into the one source buffer.
//!
//! Which payloads there are is not decided here. The pool is built from
//! [`crate::payloads::recipe_payloads`], and the writers below may only address
//! entries it produced — so the artifact carries exactly what that one walk
//! says a recipe carries, stored the way each payload's role says, and a
//! disagreement between the two is refused rather than written.

use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;

use crate::blob::{BlobPlace, StoredEncoding};
use crate::payloads::{Payload, PayloadRole};

#[path = "wire_body.rs"]
mod wire_body;
#[path = "wire_code.rs"]
mod wire_code;
#[path = "wire_io.rs"]
mod wire_io;
#[path = "wire_signature.rs"]
mod wire_signature;
use crate::algebra::CompileRequests;
use crate::{Blob, Recipe};
use wire_body::{read_body, write_body};
use wire_code::{read_request, write_request};
use wire_io::{Cursor, Writer};
use wire_signature::{SECTION_SIGNATURE, read_signature, write_signature};

const MAGIC: [u8; 4] = *b"RCPE";
/// The protocol number in the header of every artifact this build writes, and
/// the only one it reads.
///
/// It is public because a store that keeps artifacts across builds has to name
/// it: [`FORMAT.md`](../FORMAT.md) requires a persistent cache to either put
/// this in its namespace or reject entries that do not carry it, and a cache
/// that could not see it could not do either.
pub const VERSION: u16 = 13;

const SECTION_BLOBS: u32 = 1;
const SECTION_REQUESTS: u32 = 2;
const SECTION_BODY: u32 = 3;

const PAYLOAD_START: u64 = 12 + 20 * 4;

/// One section-table entry: id, offset, length.
const SECTION_ENTRY_BYTES: usize = 4 + 8 + 8;
/// The fixed part of one BLOBS entry: encoding, decoded length, digest, and
/// the payload length that follows it.
const BLOB_ENTRY_BYTES: usize = 1 + 8 + 32 + 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
enum PayloadEncoding {
    Raw = 0,
    Zstd = 1,
}

impl PayloadEncoding {
    /// How the artifact stores a payload of this role. The role decides it —
    /// see [`crate::payloads::PayloadRole::is_compressed`] — so no writer here
    /// decides it per field.
    fn of(role: PayloadRole) -> PayloadEncoding {
        if role.is_compressed() {
            PayloadEncoding::Zstd
        } else {
            PayloadEncoding::Raw
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WireError {
    BadMagic {
        found: [u8; 4],
    },
    UnsupportedVersion {
        found: u16,
    },
    MissingSection {
        id: u32,
    },
    BadTag {
        what: &'static str,
        tag: u8,
    },
    BadUtf8,
    Truncated,
    Oversize {
        what: &'static str,
        value: u64,
    },
    /// A sequence the format states as canonical arrived in some other form —
    /// out of order, or with a repeat. Two readers could disagree about what
    /// it means, so neither reads it.
    NonCanonical {
        what: &'static str,
    },
    /// The walk that says which payloads a recipe carries disagrees with the
    /// fields that reference them: a field named bytes the walk did not reach,
    /// or the walk reached bytes no field names.
    ///
    /// This is never a property of the bytes being read — it can only arise
    /// while writing, from this build holding two answers to one question. It
    /// is refused rather than papered over, because either half of the
    /// disagreement makes an artifact that is wrong rather than merely
    /// different.
    PayloadWalkDisagrees {
        what: &'static str,
    },
    BlobIndexOutOfRange {
        index: u32,
    },
    BlobDigestMismatch {
        index: u32,
    },
    BlobDecode {
        index: u32,
        error: String,
    },
    Io(String),
    /// The bytes decoded into a structure, and that structure is not a recipe:
    /// a body that contradicts the signature it arrived with, a node naming a
    /// buffer that is not there, a compile table that cycles. An encoder never
    /// produces one, so these bytes are corrupt or hostile.
    Invalid(crate::RecipeError),
}

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WireError::BadMagic { found } => {
                write!(f, "wrong magic {found:?}, expected {MAGIC:?}")
            }
            WireError::UnsupportedVersion { found } => {
                write!(f, "unsupported version {found}, this build reads {VERSION}")
            }
            WireError::MissingSection { id } => {
                write!(f, "required section {id} is missing")
            }
            WireError::BadTag { what, tag } => {
                write!(f, "unknown {what} tag {tag}")
            }
            WireError::Invalid(error) => {
                write!(f, "decoded bytes are not a valid recipe: {error}")
            }
            WireError::BadUtf8 => write!(f, "string field is not valid UTF-8"),
            WireError::Oversize { what, value } => {
                write!(
                    f,
                    "wire field {what} value {value} is larger than the format admits"
                )
            }
            WireError::NonCanonical { what } => {
                write!(f, "{what} are not in the one order this format states")
            }
            WireError::PayloadWalkDisagrees { what } => {
                write!(
                    f,
                    "the recipe's payload walk and its fields disagree: {what}"
                )
            }
            WireError::Truncated => write!(f, "buffer is truncated"),
            WireError::BlobIndexOutOfRange { index } => {
                write!(f, "blob pool index {index} is out of range")
            }
            WireError::BlobDigestMismatch { index } => {
                write!(f, "blob pool entry {index} digest does not match")
            }
            WireError::BlobDecode { index, error } => {
                write!(f, "blob pool entry {index} cannot be decoded: {error}")
            }
            WireError::Io(error) => write!(f, "artifact I/O failed: {error}"),
        }
    }
}

impl std::error::Error for WireError {}

/// Every wire count/length/index is a `u32`; this is the one place a
/// too-large `usize` turns into a loud [`WireError::Oversize`] instead of a
/// silent wraparound at encode time.
fn checked_u32(value: usize, what: &'static str) -> Result<u32, WireError> {
    u32::try_from(value).map_err(|_| WireError::Oversize {
        what,
        value: value as u64,
    })
}

/// The largest byte offset or byte count this format admits.
///
/// It is `isize::MAX`, not `usize::MAX`: no allocation may exceed it, and
/// arithmetic on a pointer past it is undefined, so a larger number cannot
/// name a byte of any buffer a caller could bind.
const MAX_NBYTES: u64 = isize::MAX as u64;

/// Every `u64` a decode turns into a byte offset or byte count comes through
/// here, so no reader writes `as usize` and hopes.
pub(super) fn decode_nbytes(value: u64, what: &'static str) -> Result<usize, WireError> {
    if value > MAX_NBYTES {
        return Err(WireError::Oversize { what, value });
    }
    Ok(value as usize)
}

/// The same bound on the way out, so the two directions admit the same values.
///
/// A writer that emitted a number no reader would accept would put a recipe on
/// the wire that cannot come back — the failure would land on whoever read it,
/// with only the bytes to explain it. It is refused here instead, where the
/// value still has a producer.
pub(super) fn encode_nbytes(value: usize, what: &'static str) -> Result<u64, WireError> {
    let value = value as u64;
    if value > MAX_NBYTES {
        return Err(WireError::Oversize { what, value });
    }
    Ok(value)
}

/// The bytes an entry holds: the payload's own, or a compressed copy this
/// writer made because the payload's role stores it compressed and it arrived
/// raw.
enum PoolBlob<'a> {
    Borrowed(&'a Blob),
    Spilled(Blob),
}

impl PoolBlob<'_> {
    fn blob(&self) -> &Blob {
        match self {
            PoolBlob::Borrowed(blob) => blob,
            PoolBlob::Spilled(blob) => blob,
        }
    }
}

struct PoolEntry<'a> {
    blob: PoolBlob<'a>,
    decoded_len: usize,
    digest: [u8; 32],
    encoding: PayloadEncoding,
    /// Whether some field in the artifact ended up naming this entry. An entry
    /// nothing names is the walk producing a payload the structure does not
    /// carry, which is the other half of a disagreement from a field naming
    /// bytes the walk missed.
    referenced: bool,
}

impl PoolEntry<'_> {
    /// The compressed run this entry stores, when it stores one. `None` for a
    /// raw entry, whose bytes are its content.
    fn compressed(&self) -> Option<&[u8]> {
        match (self.encoding, self.blob.blob().stored_encoding()) {
            (PayloadEncoding::Zstd, Some(StoredEncoding::Zstd { payload, .. })) => Some(payload),
            _ => None,
        }
    }

    fn payload(&self, index: u32) -> Result<&[u8], WireError> {
        match self.encoding {
            PayloadEncoding::Raw => {
                self.blob
                    .blob()
                    .decoded()
                    .map_err(|error| WireError::BlobDecode {
                        index,
                        error: error.to_string(),
                    })
            }
            PayloadEncoding::Zstd => match self.blob.blob().stored_encoding() {
                Some(StoredEncoding::Zstd { payload, .. }) => Ok(payload),
                _ => unreachable!("zstd pool entries retain zstd storage"),
            },
        }
    }
}

/// Which entries interning will even consider reusing, by digest.
///
/// Production buckets a payload under its own digest, so the search is over the
/// entries that could possibly be it. A test may bucket coarsely to put
/// payloads that do not collide into one bucket anyway — two contents with one
/// BLAKE3 digest cannot be constructed, and the comparison that has to resolve
/// such a pair would otherwise never run. Bucketing never changes what is
/// written: the digest an entry carries into the artifact is the payload's own.
type Bucket = fn([u8; 32]) -> [u8; 32];

/// A payload on its way into the pool: the bytes it came from, how it will be
/// stored, and how long its content is.
struct Incoming<'a> {
    blob: &'a Blob,
    encoding: PayloadEncoding,
    decoded_len: usize,
}

impl Incoming<'_> {
    /// The compressed run this payload will be stored as, when it already is
    /// one.
    fn compressed(&self) -> Option<&[u8]> {
        match (self.encoding, self.blob.stored_encoding()) {
            (PayloadEncoding::Zstd, Some(StoredEncoding::Zstd { payload, .. })) => Some(payload),
            _ => None,
        }
    }
}

/// The BLOBS section under construction.
///
/// Its contents come from [`crate::payloads::recipe_payloads`] and from nowhere
/// else: the writers below look entries up, they never add them. That is what
/// makes one walk the answer to "what does this recipe carry" for the artifact
/// as well as for detaching and for identity — a payload the walk misses
/// cannot be written, rather than being discovered a second time here and
/// quietly stored.
struct Pool<'a> {
    blobs: Vec<PoolEntry<'a>>,
    index: HashMap<[u8; 32], Vec<u32>>,
    /// Where each payload the walk produced landed, by the place it came from
    /// *and* the form its role stores it in. One `Blob` serving two roles that
    /// store it differently is two entries, and each field must find the one
    /// its own role needs.
    placed: HashMap<(BlobPlace, PayloadEncoding), u32>,
    bucket: Bucket,
}

impl Default for Pool<'_> {
    fn default() -> Self {
        Pool {
            blobs: Vec::new(),
            index: HashMap::new(),
            placed: HashMap::new(),
            bucket: |digest| digest,
        }
    }
}

impl<'a> Pool<'a> {
    /// The pool for `payloads`: one entry per distinct payload, numbered in the
    /// order the walk reaches it, each stored the way its role says.
    fn from_payloads(payloads: impl Iterator<Item = Payload<'a>>) -> Result<Pool<'a>, WireError> {
        Pool::from_payloads_bucketed(payloads, |digest| digest)
    }

    /// The same, over a stated [`Bucket`].
    fn from_payloads_bucketed(
        payloads: impl Iterator<Item = Payload<'a>>,
        bucket: Bucket,
    ) -> Result<Pool<'a>, WireError> {
        let mut pool = Pool {
            bucket,
            ..Pool::default()
        };
        for payload in payloads {
            let encoding = PayloadEncoding::of(payload.role);
            let index = pool.intern(payload.blob, encoding)?;
            pool.placed.insert((payload.blob.place(), encoding), index);
        }
        Ok(pool)
    }

    /// The entry a field's bytes were pooled into for `role`, noting that a
    /// field named it. Bytes the walk never reached have no entry, and saying
    /// so is the only honest answer: writing an index for them would mean
    /// interning behind the walk's back.
    fn index_of(&mut self, blob: &Blob, role: PayloadRole) -> Result<u32, WireError> {
        let key = (blob.place(), PayloadEncoding::of(role));
        let index = *self
            .placed
            .get(&key)
            .ok_or(WireError::PayloadWalkDisagrees {
                what: "a field names bytes the walk did not reach",
            })?;
        self.blobs[index as usize].referenced = true;
        Ok(index)
    }

    /// Whether every entry the walk produced was named by some field.
    fn every_entry_referenced(&self) -> bool {
        self.blobs.iter().all(|entry| entry.referenced)
    }

    /// Whether entry `index` is already what `incoming` would add.
    ///
    /// A digest narrows the search; it is not the bytes, and the format says so
    /// — an entry is reused on decoded equality, not on a hash agreeing. Two
    /// payloads also have to be stored the same way to share an entry: the same
    /// bytes as a compile source and as a module image are one content in two
    /// forms, and an artifact that gave a field the other form would be storing
    /// something its role does not state.
    ///
    /// Equal compressed runs settle it without decoding, because decoding is a
    /// function — which is what keeps re-pooling an artifact full of sources
    /// from decompressing one.
    fn is_the_same_payload(&self, index: u32, incoming: &Incoming<'_>) -> Result<bool, WireError> {
        let entry = &self.blobs[index as usize];
        if entry.encoding != incoming.encoding || entry.decoded_len != incoming.decoded_len {
            return Ok(false);
        }
        if let (Some(mine), Some(theirs)) = (incoming.compressed(), entry.compressed())
            && mine == theirs
        {
            return Ok(true);
        }
        fn decode(blob: &Blob, index: u32) -> Result<&[u8], WireError> {
            blob.decoded().map_err(|error| WireError::BlobDecode {
                index,
                error: error.to_string(),
            })
        }
        Ok(decode(incoming.blob, index)? == decode(entry.blob.blob(), index)?)
    }

    fn intern(&mut self, blob: &'a Blob, encoding: PayloadEncoding) -> Result<u32, WireError> {
        let index = checked_u32(self.blobs.len(), "blob pool size")?;
        let stored = blob.stored_encoding();
        let (decoded_len, digest, decoded) = match stored {
            Some(StoredEncoding::Zstd {
                decoded_len,
                digest,
                ..
            }) if encoding == PayloadEncoding::Zstd => (decoded_len, digest, None),
            _ => {
                let bytes = blob.decoded().map_err(|error| WireError::BlobDecode {
                    index,
                    error: error.to_string(),
                })?;
                (bytes.len(), *blake3::hash(bytes).as_bytes(), Some(bytes))
            }
        };
        let incoming = Incoming {
            blob,
            encoding,
            decoded_len,
        };
        let bucket = (self.bucket)(digest);
        for candidate in self.index.get(&bucket).cloned().unwrap_or_default() {
            if self.is_the_same_payload(candidate, &incoming)? {
                return Ok(candidate);
            }
        }
        let stored = match encoding {
            PayloadEncoding::Raw => PoolBlob::Borrowed(blob),
            PayloadEncoding::Zstd => match stored {
                Some(StoredEncoding::Zstd { .. }) => PoolBlob::Borrowed(blob),
                _ => PoolBlob::Spilled(
                    Blob::zstd_spilled(decoded.expect("raw source decoded")).map_err(|error| {
                        WireError::BlobDecode {
                            index,
                            error: error.to_string(),
                        }
                    })?,
                ),
            },
        };
        self.blobs.push(PoolEntry {
            blob: stored,
            decoded_len,
            digest,
            encoding,
            referenced: false,
        });
        self.index.entry(bucket).or_default().push(index);
        Ok(index)
    }
}

fn artifact_parts(recipe: &Recipe) -> Result<(Pool<'_>, Writer, Writer, Writer), WireError> {
    let mut pool = Pool::from_payloads(crate::payloads::recipe_payloads(recipe))?;
    let (requests, body, signature) = write_sections(recipe, &mut pool)?;
    Ok((pool, requests, body, signature))
}

/// The three structural sections, with every payload-bearing field naming the
/// entry `pool` already holds for it.
///
/// Taking the pool rather than filling one is the point: the fields here may
/// only *address* payloads, so a recipe whose payload walk and whose fields
/// disagree fails here, in either direction, instead of producing an artifact
/// that is missing bytes or carrying bytes nothing reads.
fn write_sections<'a>(
    recipe: &'a Recipe,
    pool: &mut Pool<'a>,
) -> Result<(Writer, Writer, Writer), WireError> {
    let mut requests = Writer::new();
    requests.count(recipe.context().len())?;
    for request in recipe.context().as_slice() {
        write_request(&mut requests, pool, request)?;
    }
    let mut body = Writer::new();
    write_body(&mut body, pool, recipe.body())?;

    if !pool.every_entry_referenced() {
        return Err(WireError::PayloadWalkDisagrees {
            what: "the walk reached bytes no field names",
        });
    }

    let mut signature = Writer::new();
    write_signature(&mut signature, recipe.contract())?;

    Ok((requests, body, signature))
}

fn artifact_lengths(
    pool: &Pool<'_>,
    requests: &Writer,
    body: &Writer,
    signature: &Writer,
) -> Result<(u64, usize), WireError> {
    let mut blobs_len = 4u64;
    for (index, blob) in pool.blobs.iter().enumerate() {
        let payload_len = blob.payload(index as u32)?.len();
        let payload_len_u32 = checked_u32(payload_len, "blob length")?;
        blobs_len = blobs_len
            .checked_add(1 + 8 + 32 + 4 + u64::from(payload_len_u32))
            .ok_or(WireError::Oversize {
                what: "BLOBS section length",
                value: u64::MAX,
            })?;
    }
    let total_u64 = PAYLOAD_START
        .checked_add(blobs_len)
        .and_then(|length| length.checked_add(requests.bytes.len() as u64))
        .and_then(|length| length.checked_add(body.bytes.len() as u64))
        .and_then(|length| length.checked_add(signature.bytes.len() as u64))
        .ok_or(WireError::Oversize {
            what: "artifact length",
            value: u64::MAX,
        })?;
    let total = usize::try_from(total_u64).map_err(|_| WireError::Oversize {
        what: "artifact length",
        value: total_u64,
    })?;
    Ok((blobs_len, total))
}

/// The table lists the four sections in id order and their payloads follow in
/// the same order, so a reader that walks the table walks the artifact forward.
/// Nothing depends on it — every section is found by its own offset — but one
/// order is simpler to state and to check than two.
fn write_artifact(
    pool: &Pool<'_>,
    requests: &Writer,
    body: &Writer,
    signature: &Writer,
    output: &mut impl Write,
) -> Result<usize, WireError> {
    let (blobs_len, total) = artifact_lengths(pool, requests, body, signature)?;

    let requests_at = PAYLOAD_START + blobs_len;
    let body_at = requests_at + requests.bytes.len() as u64;
    let signature_at = body_at + body.bytes.len() as u64;

    let mut header = Writer::new();
    header.bytes.extend_from_slice(&MAGIC);
    header.u16(VERSION);
    header.u16(0);
    header.u32(4);

    header.u32(SECTION_BLOBS);
    header.u64(PAYLOAD_START);
    header.u64(blobs_len);
    header.u32(SECTION_REQUESTS);
    header.u64(requests_at);
    header.u64(requests.bytes.len() as u64);
    header.u32(SECTION_BODY);
    header.u64(body_at);
    header.u64(body.bytes.len() as u64);
    header.u32(SECTION_SIGNATURE);
    header.u64(signature_at);
    header.u64(signature.bytes.len() as u64);
    output
        .write_all(&header.bytes)
        .map_err(|error| WireError::Io(error.to_string()))?;

    let mut count = Writer::new();
    count.count(pool.blobs.len())?;
    output
        .write_all(&count.bytes)
        .map_err(|error| WireError::Io(error.to_string()))?;
    for (index, blob) in pool.blobs.iter().enumerate() {
        let payload = blob.payload(index as u32)?;
        let mut metadata = Writer::new();
        metadata.u8(blob.encoding as u8);
        metadata.u64(encode_nbytes(blob.decoded_len, "decoded blob length")?);
        metadata.bytes.extend_from_slice(&blob.digest);
        metadata.u32(checked_u32(payload.len(), "blob length")?);
        output
            .write_all(&metadata.bytes)
            .and_then(|()| output.write_all(payload))
            .map_err(|error| WireError::Io(error.to_string()))?;
    }
    output
        .write_all(&requests.bytes)
        .and_then(|()| output.write_all(&body.bytes))
        .and_then(|()| output.write_all(&signature.bytes))
        .map_err(|error| WireError::Io(error.to_string()))?;
    Ok(total)
}

pub fn serialize(recipe: &Recipe) -> Result<Vec<u8>, WireError> {
    let (pool, requests, body, signature) = artifact_parts(recipe)?;
    let (_, total) = artifact_lengths(&pool, &requests, &body, &signature)?;
    let mut bytes = Vec::with_capacity(total);
    write_artifact(&pool, &requests, &body, &signature, &mut bytes)?;
    Ok(bytes)
}

/// Writes the artifact incrementally into an unlinked file and returns one
/// mmap-backed view. Encoded source payloads never accumulate in heap memory.
pub fn serialize_spilled(recipe: &Recipe) -> Result<Blob, WireError> {
    let (pool, requests, body, signature) = artifact_parts(recipe)?;
    let mut file = tempfile::tempfile().map_err(|error| WireError::Io(error.to_string()))?;
    write_artifact(&pool, &requests, &body, &signature, &mut file)?;
    file.flush()
        .map_err(|error| WireError::Io(error.to_string()))?;
    Blob::from_file(&file).map_err(|error| WireError::Io(error.to_string()))
}

struct SectionEntry {
    id: u32,
    offset: u64,
    length: u64,
}

fn section(bytes: &[u8], table: &[SectionEntry], id: u32) -> Result<(usize, usize), WireError> {
    let entry = table
        .iter()
        .find(|e| e.id == id)
        .ok_or(WireError::MissingSection { id })?;
    let start = entry.offset as usize;
    let end = start
        .checked_add(entry.length as usize)
        .ok_or(WireError::Truncated)?;
    if end > bytes.len() {
        return Err(WireError::Truncated);
    }
    Ok((start, end))
}

/// Decodes with one copy: the input is copied into a fresh shared buffer and
/// every payload blob is a view into it. Prefer [`deserialize_shared`] when
/// the bytes already live in a [`Blob`] (e.g. a disk-backed mapping).
pub fn deserialize(bytes: &[u8]) -> Result<Recipe, WireError> {
    deserialize_shared(Blob::from_vec(bytes.to_vec()))
}

/// Zero-copy decode: every payload blob in the result is a view into
/// `source`, which stays alive as long as any view does.
///
/// The result is a *valid* recipe: bytes that decode into a structure which is
/// not one are [`WireError::Invalid`], not a `Recipe` a caller has to remember
/// to check.
pub fn deserialize_shared(source: Blob) -> Result<Recipe, WireError> {
    let bytes: &[u8] = &source;
    let mut c = Cursor::new(bytes, 0);

    let magic: [u8; 4] = c.take(4)?.try_into().expect("take(4) yields four bytes");
    if magic != MAGIC {
        return Err(WireError::BadMagic { found: magic });
    }
    let version = c.u16()?;
    if version != VERSION {
        return Err(WireError::UnsupportedVersion { found: version });
    }
    let _reserved = c.u16()?;
    // Both counts here index directly into an allocation, so both are bounded
    // against the bytes still to come first: a table entry is 20 bytes and a
    // blob entry is at least its 45 bytes of metadata.
    let section_count = c.bounded_count(SECTION_ENTRY_BYTES)?;

    let mut table = Vec::with_capacity(section_count);
    for _ in 0..section_count {
        let id = c.u32()?;
        let offset = c.u64()?;
        let length = c.u64()?;
        table.push(SectionEntry { id, offset, length });
    }

    let (blobs_start, blobs_end) = section(bytes, &table, SECTION_BLOBS)?;
    let mut c = Cursor::new(&bytes[blobs_start..blobs_end], blobs_start);
    let n_blobs = c.bounded_count(BLOB_ENTRY_BYTES)?;
    let mut pool = Vec::with_capacity(n_blobs);
    let shared_source: Arc<dyn AsRef<[u8]> + Send + Sync> = Arc::new(source.clone());
    for index in 0..n_blobs {
        let encoding = match c.u8()? {
            0 => PayloadEncoding::Raw,
            1 => PayloadEncoding::Zstd,
            tag => {
                return Err(WireError::BadTag {
                    what: "blob encoding",
                    tag,
                });
            }
        };
        // What a payload decodes to is a byte count like every other, so it is
        // bounded like every other: a length no allocation could reach names
        // no content, and admitting one here would hand the decoder a number
        // the rest of the format has already refused.
        let decoded_len = decode_nbytes(c.u64()?, "decoded blob length")?;
        let digest: [u8; 32] = c.take(32)?.try_into().expect("32-byte digest");
        let len = c.u32()? as usize;
        let start = c.abs_pos();
        c.take(len)?;
        let payload = start..start + len;
        let blob = match encoding {
            PayloadEncoding::Raw => {
                if len != decoded_len {
                    return Err(WireError::BlobDecode {
                        index: index as u32,
                        error: format!("raw payload has {len} bytes, expected {decoded_len}"),
                    });
                }
                let blob = source.slice(payload);
                if blake3::hash(&blob).as_bytes() != &digest {
                    return Err(WireError::BlobDigestMismatch {
                        index: index as u32,
                    });
                }
                blob
            }
            PayloadEncoding::Zstd => {
                Blob::from_zstd_arc(shared_source.clone(), payload, decoded_len, digest)
            }
        };
        pool.push(blob);
    }

    let (signature_start, signature_end) = section(bytes, &table, SECTION_SIGNATURE)?;
    let mut c = Cursor::new(&bytes[signature_start..signature_end], signature_start);
    let signature = read_signature(&mut c)?;

    let (requests_start, requests_end) = section(bytes, &table, SECTION_REQUESTS)?;
    let mut c = Cursor::new(&bytes[requests_start..requests_end], requests_start);
    let compile_requests: Vec<_> = (0..c.count()?)
        .map(|_| read_request(&mut c, &pool))
        .collect::<Result<_, _>>()?;

    let (body_start, body_end) = section(bytes, &table, SECTION_BODY)?;
    let mut c = Cursor::new(&bytes[body_start..body_end], body_start);
    let inputs = signature.num_inputs();
    let outputs = signature.num_outputs();
    let body = read_body(&mut c, &pool, inputs, outputs)?;
    // Decoding is where bytes become a recipe, so it is where one that
    // contradicts itself stops. A producer validates what it finishes, so any
    // artifact reaching here that does not validate was corrupted or forged in
    // between — and every consumer downstream (a cache on disk, a Python
    // handle, a replayed capture) would otherwise take it on trust. What
    // decoding cannot recover is the statement the edges were derived from, so
    // what comes back is a recipe and not a proved one.
    Recipe::stating_edges(signature, CompileRequests::new(compile_requests), body)
        .map_err(WireError::Invalid)
}

#[cfg(test)]
#[path = "wire_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "wire_corruption_tests.rs"]
mod corruption_tests;

#[cfg(test)]
#[path = "wire_pool_tests.rs"]
mod pool_tests;
