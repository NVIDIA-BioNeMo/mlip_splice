# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Builds the MD CUDA kernels into the Python package as `splice.md._ops`.
#
# Run it from the repository root as `make -C packages/splice/python/splice/md/_ops_src`, or from
# this directory as `make`.
#
#   make                        # build for the GPU in this machine (-arch=native)
#   make ARCH=ARCH_FULL         # fat binary, every architecture we ship for
#   make PYTHON=/path/to/python # build against a specific interpreter
#   make clean
#
# Requires nvcc and pybind11 (`pip install pybind11`). This is a separate step
# from `maturin develop`, which builds the Rust side; see README.md.
#
# PYTHON must be the interpreter you will import from: it supplies both the
# headers and the extension suffix. Inside a virtualenv, `python3` on PATH is
# usually already the right one; in a container it often is not, and the
# mismatch shows up as a module that will not import rather than as a build
# failure.

CUDA_HOME    ?= $(shell dirname $(shell dirname $(shell which nvcc)))
PYTHON       ?= python3

ARCH_FULL     = -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_100,code=sm_100 \
  -gencode arch=compute_120,code=sm_120 \
  -gencode arch=compute_120,code=compute_120

ARCH_NATIVE   = -arch=native
ARCH         ?= ARCH_NATIVE

NVCC          = nvcc $($(ARCH)) -std=c++17 -O3 -use_fast_math -Xcompiler -fPIC -lineinfo
PYBIND_INC    = $(shell $(PYTHON) -m pybind11 --includes)
BUILD_OPTS    = -I$(CUDA_HOME)/include -Icommon $(PYBIND_INC)
LINK_OPTS     = -L$(CUDA_HOME)/lib64 -lcudart -lcurand

TARGET_SUFFIX = $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
TARGET        = ../_ops$(TARGET_SUFFIX)

SOURCES       = $(wildcard md/*.cu) common/workspace.cu
OBJECTS       = $(addprefix .obj/,$(SOURCES:.cu=.o))
DEPENDS       = $(addprefix .obj/,$(SOURCES:.cu=.d))

default: check $(TARGET)

# Fail loudly here rather than emit an extension that cannot be imported.
check:
	@$(PYTHON) -c "import pybind11" 2>/dev/null || \
	  { echo "error: $(PYTHON) cannot import pybind11 (pip install pybind11),"; \
	    echo "       or pass PYTHON=<the interpreter you will import from>."; exit 1; }

.obj/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(BUILD_OPTS) -MMD -MP -c $< -o $@

$(TARGET): $(OBJECTS)
	@mkdir -p $(dir $(TARGET))
	$(NVCC) -shared $(BUILD_OPTS) -o $(TARGET) $(OBJECTS) $(LINK_OPTS)
	@echo "built $(TARGET)"

clean:
	rm -rf .obj $(TARGET)

.PHONY: default check clean

-include $(DEPENDS)
