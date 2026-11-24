# Third-Party Notices

This project includes third-party open source software. Each component is listed
below with its copyright holder and license.

## OpenXLA

```
OpenXLA
Copyright 2023 The OpenXLA Authors
Licensed under the Apache License, Version 2.0
```

Vendored file: `crates/integrations/xla_ffi/sys/vendor/c_api.h`, an unmodified
copy of the XLA FFI C API header, included so that this project can be called
from XLA without depending on an XLA source tree at build time. The file retains
its original copyright notice and carries no NVIDIA copyright line.

The upstream header was distributed without a `NOTICE` file, so the conditional
attribution requirement of Apache 2.0 section 4(d) is not triggered.

## Build-time dependencies

No other third-party source code is included in this repository. The Rust and
Python packages this project builds against are declared in the root `Cargo.toml`
(`[workspace.dependencies]`) and in each package's `pyproject.toml`
(`packages/*/pyproject.toml`), pinned in `Cargo.lock` and `uv.lock`, and are
fetched from upstream at build time; they are not redistributed here.
