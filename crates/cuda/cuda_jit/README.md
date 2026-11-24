# cuda_jit

NVRTC + nvJitLink, as one stateless value.

`CudaJit::load(CudaJitConfig { lib_dirs, include_dirs })` `dlopen`s `libnvrtc`
and `libnvJitLink`. `to_cubin` / `to_ltoir` then compile a CUDA source buffer to
a cubin (or relocatable LTO-IR), fusing extra LTO-IR blobs (e.g. cuFFTDx) when
given. Plaintext CUDA source is compiled through NVRTC.

No globals: the caller owns the `CudaJit` and its search paths. `cubin_key` is
the content-addressed cache key, but caching is `blob_cache`'s job, device
queries and loading are `cuda_driver`'s, and launching is `kiln`'s — this crate
only turns bytes into bytes.
