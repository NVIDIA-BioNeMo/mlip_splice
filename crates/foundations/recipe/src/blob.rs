// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared immutable byte storage: one owner per byte.
//!
//! A [`Blob`] is a cheaply-clonable view into reference-counted bytes. A
//! decoded recipe's payloads (cubins, workspace inits) are views into the one
//! buffer it was decoded from — heap or disk-backed — so cloning recipes and
//! candidates never copies payload bytes.

use std::io::{Seek, Write};
use std::ops::{Deref, Range};
use std::sync::{Arc, OnceLock};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlobDecodeError {
    Zstd(String),
    Length {
        expected: usize,
        found: usize,
    },
    /// The payload goes on producing bytes past the length it states.
    ///
    /// How many more is deliberately not measured: reading a hostile stream to
    /// its end in order to say how long it is, is the very thing refusing it
    /// avoids. What is established is that it exceeds `expected`, and that is
    /// what this says.
    Overlong {
        expected: usize,
    },
    Digest,
}

impl std::fmt::Display for BlobDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlobDecodeError::Zstd(error) => write!(f, "zstd decode failed: {error}"),
            BlobDecodeError::Length { expected, found } => {
                write!(f, "decoded blob has {found} bytes, expected {expected}")
            }
            BlobDecodeError::Overlong { expected } => {
                write!(
                    f,
                    "decoded blob is longer than the {expected} bytes it states"
                )
            }
            BlobDecodeError::Digest => write!(f, "decoded blob digest does not match"),
        }
    }
}

/// How much of an encoded payload is expanded at a time.
///
/// The working buffer is this and nothing more, whatever a payload states or
/// produces, so what a hostile artifact can make a reader allocate up front is
/// a constant rather than a number it chose.
const EXPANSION_CHUNK: usize = 64 * 1024;

/// Expand `payload` and hand each chunk to `take`, refusing a stream that runs
/// past the length it states.
///
/// One pass, one bounded buffer: the decoder is never asked for more than the
/// claim still admits plus the one byte that would disprove it, so a payload
/// that keeps producing is stopped at the boundary instead of after it. A
/// stream that stops short is refused too — the claim is a statement about the
/// whole content, and half of it is not that content.
fn expand_encoded(
    payload: &[u8],
    decoded_len: usize,
    mut take: impl FnMut(&[u8]),
) -> Result<(), BlobDecodeError> {
    use std::io::Read;

    let zstd = |error: std::io::Error| BlobDecodeError::Zstd(error.to_string());
    let mut decoder = zstd::stream::read::Decoder::new(payload).map_err(zstd)?;
    let mut chunk = vec![0u8; EXPANSION_CHUNK];
    let mut seen = 0usize;
    loop {
        // `seen` never passes `decoded_len` — the check below is what stops it
        // — so this subtraction is total, and the one-byte overrun probe
        // saturates rather than wrapping: a payload is free to state a length
        // of `usize::MAX`, and stating one must not be how it decides the
        // arithmetic here.
        let admits = decoded_len - seen;
        let want = admits.saturating_add(1).min(chunk.len());
        let read = decoder.read(&mut chunk[..want]).map_err(zstd)?;
        if read == 0 {
            break;
        }
        if read > admits {
            return Err(BlobDecodeError::Overlong {
                expected: decoded_len,
            });
        }
        take(&chunk[..read]);
        seen += read;
    }
    if seen != decoded_len {
        return Err(BlobDecodeError::Length {
            expected: decoded_len,
            found: seen,
        });
    }
    Ok(())
}

/// The whole of an encoded payload, expanded under the length it states and
/// checked against the digest it states.
///
/// The one place bytes come out of encoded storage, so what makes a payload
/// acceptable is one answer rather than one per caller.
fn expanded(
    payload: &[u8],
    decoded_len: usize,
    digest: &[u8; 32],
) -> Result<Vec<u8>, BlobDecodeError> {
    let mut bytes = Vec::with_capacity(decoded_len.min(EXPANSION_CHUNK));
    expand_encoded(payload, decoded_len, |chunk| bytes.extend_from_slice(chunk))?;
    if blake3::hash(&bytes).as_bytes() != digest {
        return Err(BlobDecodeError::Digest);
    }
    Ok(bytes)
}

impl std::error::Error for BlobDecodeError {}

enum Storage {
    Raw(Arc<dyn AsRef<[u8]> + Send + Sync>),
    Zstd {
        source: Arc<dyn AsRef<[u8]> + Send + Sync>,
        payload: Range<usize>,
        decoded_len: usize,
        digest: [u8; 32],
        decoded: OnceLock<Result<Vec<u8>, BlobDecodeError>>,
    },
}

#[derive(Clone, Copy)]
pub(crate) enum StoredEncoding<'a> {
    Raw,
    Zstd {
        payload: &'a [u8],
        decoded_len: usize,
        digest: [u8; 32],
    },
}

/// Where a [`Blob`]'s bytes live: the storage it views, and the part of it.
///
/// Two clones of one blob are one place; two blobs holding equal bytes in
/// separate allocations are two.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct BlobPlace {
    storage: *const (),
    start: usize,
    end: usize,
}

#[derive(Clone)]
pub struct Blob {
    storage: Arc<Storage>,
    range: Range<usize>,
}

impl Blob {
    pub fn len(&self) -> usize {
        self.range.len()
    }

    pub fn is_empty(&self) -> bool {
        self.range.is_empty()
    }

    pub fn from_arc(src: Arc<dyn AsRef<[u8]> + Send + Sync>) -> Blob {
        let len = (*src).as_ref().len();
        Blob {
            storage: Arc::new(Storage::Raw(src)),
            range: 0..len,
        }
    }

    pub(crate) fn from_zstd_arc(
        source: Arc<dyn AsRef<[u8]> + Send + Sync>,
        payload: Range<usize>,
        decoded_len: usize,
        digest: [u8; 32],
    ) -> Blob {
        Blob {
            storage: Arc::new(Storage::Zstd {
                source,
                payload,
                decoded_len,
                digest,
                decoded: OnceLock::new(),
            }),
            range: 0..decoded_len,
        }
    }

    pub fn from_vec(bytes: Vec<u8>) -> Blob {
        Blob::from_arc(Arc::new(bytes))
    }

    pub(crate) fn from_file(file: &std::fs::File) -> std::io::Result<Blob> {
        let map = unsafe { memmap2::Mmap::map(file) }?;
        Ok(Blob::from_arc(Arc::new(map)))
    }

    /// Compresses into the heap and returns a view that decodes on demand.
    ///
    /// The compressed bytes are the blob's whole storage, so a payload that is
    /// born compressed stays that way: nothing here retains the expanded form,
    /// and serialization is not the first compression boundary.
    pub fn zstd(bytes: &[u8]) -> std::io::Result<Blob> {
        let encoded = zstd::stream::encode_all(bytes, 3)?;
        let encoded_len = encoded.len();
        Ok(Blob {
            storage: Arc::new(Storage::Zstd {
                source: Arc::new(encoded),
                payload: 0..encoded_len,
                decoded_len: bytes.len(),
                digest: *blake3::hash(bytes).as_bytes(),
                decoded: OnceLock::new(),
            }),
            range: 0..bytes.len(),
        })
    }

    /// Compresses into an unlinked temporary file and returns a lazy decoded
    /// view backed by that file's page cache.
    pub fn zstd_spilled(bytes: &[u8]) -> std::io::Result<Blob> {
        Self::zstd_spilled_many(std::iter::once(bytes)).map(|mut blobs| blobs.remove(0))
    }

    pub fn zstd_spilled_many<'a>(
        sources: impl IntoIterator<Item = &'a [u8]>,
    ) -> std::io::Result<Vec<Blob>> {
        let mut file = tempfile::tempfile()?;
        let mut metadata = Vec::new();
        for bytes in sources {
            let start = file.stream_position()? as usize;
            let mut encoder = zstd::stream::write::Encoder::new(file, 3)?;
            encoder.write_all(bytes)?;
            file = encoder.finish()?;
            let end = file.stream_position()? as usize;
            metadata.push((start..end, bytes.len(), *blake3::hash(bytes).as_bytes()));
        }
        if metadata.is_empty() {
            return Ok(Vec::new());
        }
        let map = unsafe { memmap2::Mmap::map(&file) }?;
        let source: Arc<dyn AsRef<[u8]> + Send + Sync> = Arc::new(map);
        Ok(metadata
            .into_iter()
            .map(|(payload, decoded_len, digest)| {
                Blob::from_zstd_arc(source.clone(), payload, decoded_len, digest)
            })
            .collect())
    }

    /// Which bytes this is as a *place* rather than as content.
    ///
    /// It is what lets a consumer that was handed a payload by one walk find
    /// the entry that walk made for it, without hashing the bytes a second
    /// time — content-equal payloads still share that entry, because the pool
    /// they went into deduplicated them by digest when it was built.
    pub(crate) fn place(&self) -> BlobPlace {
        BlobPlace {
            storage: Arc::as_ptr(&self.storage) as *const (),
            start: self.range.start,
            end: self.range.end,
        }
    }

    pub(crate) fn stored_encoding(&self) -> Option<StoredEncoding<'_>> {
        match self.storage.as_ref() {
            Storage::Raw(_) => Some(StoredEncoding::Raw),
            Storage::Zstd {
                source,
                payload,
                decoded_len,
                digest,
                ..
            } if self.range == (0..*decoded_len) => Some(StoredEncoding::Zstd {
                payload: &source.as_ref().as_ref()[payload.clone()],
                decoded_len: *decoded_len,
                digest: *digest,
            }),
            Storage::Zstd { .. } => None,
        }
    }

    /// A sub-view; `range` is relative to `self`.
    pub fn slice(&self, range: Range<usize>) -> Blob {
        assert!(
            range.start <= range.end && range.end <= self.range.len(),
            "blob slice {range:?} out of bounds for length {}",
            self.range.len()
        );
        Blob {
            storage: self.storage.clone(),
            range: self.range.start + range.start..self.range.start + range.end,
        }
    }

    pub fn decoded(&self) -> Result<&[u8], BlobDecodeError> {
        let bytes = match self.storage.as_ref() {
            Storage::Raw(source) => source.as_ref().as_ref(),
            Storage::Zstd {
                source,
                payload,
                decoded_len,
                digest,
                decoded,
            } => decoded
                .get_or_init(|| {
                    expanded(
                        &source.as_ref().as_ref()[payload.clone()],
                        *decoded_len,
                        digest,
                    )
                })
                .as_ref()
                .map_err(Clone::clone)?,
        };
        Ok(&bytes[self.range.clone()])
    }

    /// Decodes for one operation without retaining the expanded bytes. Raw
    /// storage remains zero-copy; encoded storage is released when `use_bytes`
    /// returns.
    pub fn with_decoded<R>(
        &self,
        use_bytes: impl FnOnce(&[u8]) -> R,
    ) -> Result<R, BlobDecodeError> {
        match self.storage.as_ref() {
            Storage::Raw(source) => Ok(use_bytes(&source.as_ref().as_ref()[self.range.clone()])),
            Storage::Zstd {
                source,
                payload,
                decoded_len,
                digest,
                ..
            } => {
                let bytes = expanded(
                    &source.as_ref().as_ref()[payload.clone()],
                    *decoded_len,
                    digest,
                )?;
                Ok(use_bytes(&bytes[self.range.clone()]))
            }
        }
    }

    /// The same over several blobs at once, for an operation that needs all of
    /// them decoded together — assembling one compile unit out of the modules
    /// it links, say.
    ///
    /// Every expansion lives inside this call and nowhere after it, exactly as
    /// [`Blob::with_decoded`]'s does, so a pool of compressed sources stays
    /// compressed however many of them one operation reads.
    pub fn with_all_decoded<R>(
        blobs: &[Blob],
        use_bytes: impl FnOnce(&[&[u8]]) -> R,
    ) -> Result<R, BlobDecodeError> {
        // A compressed blob is expanded into a buffer this call owns — the
        // same allocation `with_decoded` makes, held here so every input can
        // be read at once and dropped when the call returns. Raw storage is
        // borrowed where it lies and expands nothing, so the cost is the
        // compressed inputs' decoded bytes and one pointer per blob.
        let mut expanded: Vec<Option<Vec<u8>>> = Vec::with_capacity(blobs.len());
        for blob in blobs {
            expanded.push(match blob.storage.as_ref() {
                Storage::Raw(_) => None,
                Storage::Zstd { .. } => Some(blob.with_decoded(<[u8]>::to_vec)?),
            });
        }
        let decoded: Vec<&[u8]> = blobs
            .iter()
            .zip(&expanded)
            .map(|(blob, expanded)| match expanded {
                Some(bytes) => Ok(bytes.as_slice()),
                None => blob.decoded(),
            })
            .collect::<Result<_, _>>()?;
        Ok(use_bytes(&decoded))
    }

    /// Whether two blobs hold the same bytes — proved, never assumed.
    ///
    /// This exists because [`Blob::claimed_digest`] answers a different
    /// question: for an encoded blob it returns the digest its artifact
    /// *claims*, which nothing has checked until the bytes are expanded. A
    /// consumer that dropped, interned or substituted one payload for another
    /// on that claim would act on bytes it never read — so every such consumer
    /// asks here, and a digest is left to do what it can do, which is narrow
    /// the candidates worth asking about.
    ///
    /// One blob viewed twice is settled without reading anything, and so are
    /// two blobs whose whole stored encoding agrees — same compressed run,
    /// same stated length, same stated digest — since those are two views of
    /// one stored payload and cannot turn out differently from each other.
    /// Anything else is expanded and compared, which is also what checks that
    /// each holds the bytes it claims. Nothing is retained, so comparing two
    /// compressed sources leaves them compressed.
    pub fn same_content(&self, other: &Blob) -> Result<bool, BlobDecodeError> {
        if self.place() == other.place() {
            return Ok(true);
        }
        if let (Some(left), Some(right)) = (self.stored_zstd(), other.stored_zstd())
            && left == right
        {
            return Ok(true);
        }
        self.with_decoded(|left| other.with_decoded(|right| left == right))?
    }

    /// The compressed bytes themselves, with what they decode to, for a writer
    /// that carries this payload somewhere else.
    ///
    /// It is what keeps serialization from being a second compression boundary:
    /// a blob that is already compressed is written as it stands, and nothing
    /// expands it on the way out.
    pub fn stored_zstd(&self) -> Option<(&[u8], usize, [u8; 32])> {
        match self.stored_encoding()? {
            StoredEncoding::Raw => None,
            StoredEncoding::Zstd {
                payload,
                decoded_len,
                digest,
            } => Some((payload, decoded_len, digest)),
        }
    }

    /// A compressed payload read back from somewhere else, believed only as far
    /// as its digest: nothing is decoded here, and a payload that does not hash
    /// to what is stated becomes a structured error at the first decode.
    pub fn from_zstd(payload: Vec<u8>, decoded_len: usize, digest: [u8; 32]) -> Blob {
        let encoded_len = payload.len();
        Blob::from_zstd_arc(Arc::new(payload), 0..encoded_len, decoded_len, digest)
    }

    /// The digest of the bytes this blob actually holds.
    ///
    /// Encoded storage is expanded and hashed rather than believed — and the
    /// expansion is where a payload that does not hash to what it states
    /// becomes an error, so this either answers what the content *is* or says
    /// the blob is not what it claims. Two blobs that answer differently are
    /// different content, and that is what makes this the digest an identity
    /// may be built from.
    ///
    /// Nothing is expanded into memory: the payload streams through in
    /// fixed-size pieces and is hashed as it goes, so proving a
    /// gigabyte-scale recipe costs one pass and a constant buffer whatever its
    /// payloads state or produce. It does cost that pass, which
    /// [`Blob::claimed_digest`] is exactly the way to avoid wherever a digest
    /// is only narrowing candidates.
    pub fn content_digest(&self) -> Result<[u8; 32], BlobDecodeError> {
        let (source, payload, decoded_len, digest) = match self.storage.as_ref() {
            Storage::Raw(source) => {
                let bytes = &source.as_ref().as_ref()[self.range.clone()];
                return Ok(*blake3::hash(bytes).as_bytes());
            }
            Storage::Zstd {
                source,
                payload,
                decoded_len,
                digest,
                ..
            } => (source, payload, *decoded_len, digest),
        };
        // The claim covers the whole stored content, so the whole of it is
        // hashed to check it. A sub-view's own digest is a second hasher over
        // the part inside its range — a different value, not the same one
        // twice — and a whole view needs no second hasher at all.
        let mut whole = blake3::Hasher::new();
        let mut view = (self.range != (0..decoded_len)).then(blake3::Hasher::new);
        let mut seen = 0usize;
        expand_encoded(
            &source.as_ref().as_ref()[payload.clone()],
            decoded_len,
            |chunk| {
                whole.update(chunk);
                if let Some(view) = &mut view {
                    let start = self.range.start.saturating_sub(seen).min(chunk.len());
                    let end = self.range.end.saturating_sub(seen).min(chunk.len());
                    if start < end {
                        view.update(&chunk[start..end]);
                    }
                }
                seen += chunk.len();
            },
        )?;
        if whole.finalize().as_bytes() != digest {
            return Err(BlobDecodeError::Digest);
        }
        Ok(*view.unwrap_or(whole).finalize().as_bytes())
    }

    /// The digest this blob's artifact *states* for its content.
    ///
    /// For raw storage that is the content's own digest, because the bytes are
    /// there to hash. For encoded storage it is the number the wire carries,
    /// which nothing has checked: an artifact is free to state a digest its
    /// payload does not have, and only expanding it finds out.
    ///
    /// So this is a lookup key and never an identity. It may narrow which
    /// candidates are worth comparing; whether two blobs are the same content
    /// is [`Blob::same_content`]'s to answer, and what a blob *is* is
    /// [`Blob::content_digest`]'s.
    pub fn claimed_digest(&self) -> Result<[u8; 32], BlobDecodeError> {
        match self.stored_encoding() {
            Some(StoredEncoding::Zstd { digest, .. }) => Ok(digest),
            _ => self.content_digest(),
        }
    }

    /// A fresh self-owned copy of just these bytes, severed from the shared
    /// source — so a small view stops pinning a large backing buffer.
    pub fn detached(&self) -> Blob {
        match self.storage.as_ref() {
            Storage::Raw(source) => {
                Blob::from_vec(source.as_ref().as_ref()[self.range.clone()].to_vec())
            }
            Storage::Zstd {
                source,
                payload,
                decoded_len,
                digest,
                ..
            } => {
                let encoded = source.as_ref().as_ref()[payload.clone()].to_vec();
                let encoded_len = encoded.len();
                Blob {
                    storage: Arc::new(Storage::Zstd {
                        source: Arc::new(encoded),
                        payload: 0..encoded_len,
                        decoded_len: *decoded_len,
                        digest: *digest,
                        decoded: OnceLock::new(),
                    }),
                    range: self.range.clone(),
                }
            }
        }
    }

    /// Copies `bytes` into an unlinked temporary file and maps it, so the
    /// payload lives in evictable page cache instead of anonymous heap.
    pub fn spilled(bytes: &[u8]) -> std::io::Result<Blob> {
        Self::spilled_with_suffix(bytes, &[])
    }

    pub fn spilled_with_suffix(bytes: &[u8], suffix: &[u8]) -> std::io::Result<Blob> {
        if bytes.is_empty() && suffix.is_empty() {
            return Ok(Blob::default());
        }
        let mut file = tempfile::tempfile()?;
        file.write_all(bytes)?;
        file.write_all(suffix)?;
        let map = unsafe { memmap2::Mmap::map(&file) }?;
        Ok(Blob::from_arc(Arc::new(map)))
    }
}

impl Deref for Blob {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.decoded().expect("invalid encoded recipe blob")
    }
}

impl AsRef<[u8]> for Blob {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

impl Default for Blob {
    fn default() -> Blob {
        Blob::from_arc(Arc::new([]))
    }
}

impl From<Vec<u8>> for Blob {
    fn from(bytes: Vec<u8>) -> Blob {
        Blob::from_vec(bytes)
    }
}

impl PartialEq for Blob {
    fn eq(&self, other: &Blob) -> bool {
        match (self.decoded(), other.decoded()) {
            (Ok(left), Ok(right)) => left == right,
            _ => false,
        }
    }
}

impl Eq for Blob {}

impl std::fmt::Debug for Blob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let encoding = match self.storage.as_ref() {
            Storage::Raw(_) => "raw",
            Storage::Zstd { .. } => "zstd",
        };
        write!(f, "Blob({} bytes, {encoding})", self.range.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What `same_content` may settle without reading: one blob viewed twice,
    /// and two views of one stored payload. A payload whose *stated* length or
    /// digest differs is a different claim about the same run, and only one of
    /// the two can be right — so it is expanded, and the wrong one fails.
    #[test]
    fn a_shared_payload_under_a_different_claim_is_not_the_same_content() {
        let bytes = vec![b'p'; 4096];
        let honest = Blob::zstd(&bytes).unwrap();
        let Some(StoredEncoding::Zstd {
            payload,
            decoded_len,
            digest,
        }) = honest.stored_encoding()
        else {
            panic!("expected zstd storage")
        };
        let run: Arc<dyn AsRef<[u8]> + Send + Sync> = Arc::new(payload.to_vec());
        let claiming = |decoded_len, digest| {
            Blob::from_zstd_arc(
                run.clone(),
                0..run.as_ref().as_ref().len(),
                decoded_len,
                digest,
            )
        };

        // The same run under the same claim: two views of one payload.
        assert_eq!(
            claiming(decoded_len, digest).same_content(&claiming(decoded_len, digest)),
            Ok(true)
        );
        assert_eq!(honest.same_content(&honest), Ok(true), "one blob is itself");

        // The same run under a forged length, and under a forged digest. Each
        // is expanded rather than taken for its twin, and the lie is what
        // comes back.
        assert_eq!(
            claiming(decoded_len, digest).same_content(&claiming(decoded_len - 1, digest)),
            Err(BlobDecodeError::Overlong {
                expected: decoded_len - 1,
            })
        );
        assert_eq!(
            claiming(decoded_len, digest).same_content(&claiming(decoded_len, [0; 32])),
            Err(BlobDecodeError::Digest)
        );
    }

    /// A payload that expands far past the length it states is refused at the
    /// boundary, not after it.
    ///
    /// `Overlong` is reachable only from a reader that stopped: one that
    /// expanded the stream to compare lengths afterwards would answer
    /// `Length` — and would have allocated everything the payload chose to
    /// produce first, which is the whole of the attack.
    #[test]
    fn a_payload_that_runs_past_its_claim_is_stopped_at_the_claim() {
        // Eight mebibytes of one byte: a few hundred stored, and what a
        // reader that believed the stream would have to hold.
        let bomb = vec![b'\0'; 8 * 1024 * 1024];
        let honest = Blob::zstd(&bomb).unwrap();
        let Some(StoredEncoding::Zstd { payload, .. }) = honest.stored_encoding() else {
            panic!("expected zstd storage")
        };
        assert!(payload.len() < 4096, "the bomb is small as stored");
        let run: Arc<dyn AsRef<[u8]> + Send + Sync> = Arc::new(payload.to_vec());

        let claiming_little = Blob::from_zstd_arc(
            run.clone(),
            0..run.as_ref().as_ref().len(),
            16,
            *blake3::hash(&bomb[..16]).as_bytes(),
        );
        assert_eq!(
            claiming_little.content_digest(),
            Err(BlobDecodeError::Overlong { expected: 16 })
        );
        assert_eq!(
            claiming_little.with_decoded(<[u8]>::to_vec),
            Err(BlobDecodeError::Overlong { expected: 16 })
        );

        // A length no allocation could reach is a number a payload is free to
        // state, and stating one decides nothing about the arithmetic that
        // reads it: the probe for one byte past the claim saturates instead of
        // wrapping, so this is refused for what it is rather than panicking.
        let vast = Blob::from_zstd(
            payload.to_vec(),
            usize::MAX,
            *blake3::hash(&bomb).as_bytes(),
        );
        assert_eq!(
            vast.content_digest(),
            Err(BlobDecodeError::Length {
                expected: usize::MAX,
                found: bomb.len(),
            })
        );

        // A stream that stops short of its claim is refused too, and there the
        // count is a fact the reader really established.
        let truncated = Blob::from_zstd_arc(
            run,
            0..payload.len(),
            bomb.len() + 1,
            *blake3::hash(&bomb).as_bytes(),
        );
        assert_eq!(
            truncated.content_digest(),
            Err(BlobDecodeError::Length {
                expected: bomb.len() + 1,
                found: bomb.len(),
            })
        );
    }

    #[test]
    fn slice_views_share_storage() {
        let blob = Blob::from_vec((0u8..100).collect());
        let view = blob.slice(10..20);
        assert_eq!(&*view, &(10u8..20).collect::<Vec<u8>>()[..]);
        let sub = view.slice(5..10);
        assert_eq!(&*sub, &(15u8..20).collect::<Vec<u8>>()[..]);
    }

    #[test]
    fn detached_equals_but_does_not_share() {
        let blob = Blob::from_vec(vec![1, 2, 3, 4]);
        let view = blob.slice(1..3);
        let own = view.detached();
        assert_eq!(own, view);
        assert_eq!(&*own, &[2, 3]);
    }

    #[test]
    fn detached_encoded_blob_stays_encoded() {
        let raw = vec![b'x'; 4096];
        let blob = Blob::zstd_spilled(&raw).unwrap();
        let detached = blob.detached();
        assert!(matches!(
            detached.stored_encoding(),
            Some(StoredEncoding::Zstd { .. })
        ));
        assert_eq!(detached.decoded().unwrap(), raw);
    }

    #[test]
    fn compressed_batch_shares_storage_and_decodes_independently() {
        let first = vec![b'a'; 4096];
        let second = vec![b'b'; 8192];
        let blobs = Blob::zstd_spilled_many([first.as_slice(), second.as_slice()]).unwrap();
        assert_eq!(blobs[0].decoded().unwrap(), first);
        assert_eq!(blobs[1].decoded().unwrap(), second);
        let (Storage::Zstd { source: left, .. }, Storage::Zstd { source: right, .. }) =
            (blobs[0].storage.as_ref(), blobs[1].storage.as_ref())
        else {
            panic!("batch blobs are not encoded")
        };
        assert!(Arc::ptr_eq(left, right));
    }

    #[test]
    fn spilled_round_trips() {
        let bytes: Vec<u8> = (0..=255)
            .cycle()
            .take(10_000)
            .map(|b: u16| b as u8)
            .collect();
        let blob = Blob::spilled(&bytes).unwrap();
        assert_eq!(&*blob, &bytes[..]);
        assert_eq!(&*Blob::spilled(&[]).unwrap(), &[] as &[u8]);
    }

    #[test]
    fn zstd_spilled_is_lazy_and_round_trips() {
        let bytes = vec![b'x'; 100_000];
        let blob = Blob::zstd_spilled(&bytes).unwrap();
        let Some(StoredEncoding::Zstd {
            payload,
            decoded_len,
            digest,
        }) = blob.stored_encoding()
        else {
            panic!("expected zstd storage");
        };
        assert!(payload.len() < bytes.len());
        assert_eq!(decoded_len, bytes.len());
        assert_eq!(digest, *blake3::hash(&bytes).as_bytes());
        assert_eq!(&*blob, &bytes);
    }

    /// The heap constructor stores exactly what the spilled one does, so a
    /// payload born compressed in memory serializes as an already-compressed
    /// stage rather than being compressed again on the way out.
    #[test]
    fn in_memory_zstd_is_stored_compressed_and_round_trips() {
        let bytes = vec![b'p'; 100_000];
        let blob = Blob::zstd(&bytes).unwrap();
        let Some(StoredEncoding::Zstd {
            payload,
            decoded_len,
            digest,
        }) = blob.stored_encoding()
        else {
            panic!("expected zstd storage");
        };
        assert!(payload.len() < bytes.len());
        assert_eq!(decoded_len, bytes.len());
        assert_eq!(digest, *blake3::hash(&bytes).as_bytes());
        assert_eq!(blob.with_decoded(<[u8]>::to_vec).unwrap(), bytes);
        assert_eq!(
            blob.content_digest().unwrap(),
            *blake3::hash(&bytes).as_bytes()
        );
        assert_eq!(
            blob.claimed_digest().unwrap(),
            blob.content_digest().unwrap()
        );
    }

    /// Empty is a payload like any other: it compresses, decodes back to
    /// nothing, and is not confused with an absent blob.
    #[test]
    fn in_memory_zstd_of_nothing_is_nothing() {
        let blob = Blob::zstd(&[]).unwrap();
        assert!(blob.is_empty());
        assert_eq!(blob.with_decoded(<[u8]>::to_vec).unwrap(), Vec::<u8>::new());
    }

    /// A corrupt payload is a structured error at the decode, not a panic
    /// through `Deref`.
    #[test]
    fn a_corrupt_in_memory_payload_is_a_structured_error() {
        let blob = Blob::zstd(&vec![b'q'; 4096]).unwrap();
        let Some(StoredEncoding::Zstd { payload, .. }) = blob.stored_encoding() else {
            panic!("expected zstd storage");
        };
        let mut broken = payload.to_vec();
        let last = broken.len() - 1;
        broken[last] ^= 0xff;
        let broken_len = broken.len();
        let corrupt = Blob::from_zstd_arc(Arc::new(broken), 0..broken_len, 4096, [0u8; 32]);
        assert!(corrupt.with_decoded(<[u8]>::to_vec).is_err());
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn slice_out_of_bounds_panics() {
        Blob::from_vec(vec![0; 4]).slice(2..6);
    }
}
