CC ?= cc
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
else
NATIVE_CPU_FLAG ?= -march=native
endif

DEBUG_FLAGS ?= -g
CFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -fobjc-arc

LDLIBS ?= -lm -pthread
METAL_SRCS := $(wildcard metal/*.metal)
CUDA_EXTRA_BINS :=

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
CORE_OBJS = ds4.o ds4_image.o ds4_distributed.o ds4_metal.o
CPU_CORE_OBJS = ds4_cpu.o ds4_image.o ds4_distributed.o
else
CFLAGS += -D_GNU_SOURCE -fno-finite-math-only
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
# Persisted CUDA build configuration: the cuda-spark / cuda-generic / cuda
# targets record their flags here and every invocation includes the record,
# so a stale CUDA object (e.g. after a sync touches ds4_cuda.cu) can only be
# recompiled with the configuration the rest of the tree was built with --
# never silently with the bare nvcc defaults (compute_75 PTX, JIT'd onto the
# device with older codepaths: slower, and a different numeric profile than
# the arch-native SASS the tree's other objects carry). Command-line
# variables still override; switch configurations by running a cuda-*
# target; survives make clean deliberately.
-include .ds4-cuda-config.mk
ifneq ($(strip $(CUDA_ARCH)),)
ifeq ($(strip $(CUDA_ARCH)),sm_121)
# GB10: the v0.5 mxf4 block-scale MMA (indexer rr-selector) needs the
# arch-SPECIFIC target.  -arch=sm_121a alone silently emits .target sm_121
# and ptxas rejects the MMA, so the gencode pair is mandatory; sm_121a
# SASS runs on every sm_121 device.  DS4_CUDA_HAVE_MXF4 gates the kernels
# AND the host engage path so non-121a builds stay coherent.
NVCC_ARCH_FLAGS := -gencode arch=compute_121a,code=sm_121a -DDS4_CUDA_HAVE_MXF4=1
else
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
endif
NVCC_EXTRA_FLAGS ?=
NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math -std=c++17 $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread $(NVCC_EXTRA_FLAGS)
# deepmem lite-2 (plumbing deleted in D3-3): DS4_CUDA_SPARK_HBM_CACHE is
# retired.  Startup weight promotion is compiled unconditionally and gated
# at runtime (integrated devices only; policy knob DS4_WEIGHT_RESIDENCY,
# legacy opt-out DS4_CUDA_NO_HBM_CACHE), so cuda-spark is purely an
# arch-selection alias for CUDA_ARCH=sm_121 and the 08-05 installer
# plan-overcommit class (forum 378855/65) cannot be built.
# Include path so cuda/mmq/*.cu can find its sibling vendored headers and
# the ds4_ggml_stubs shim. The redirected ggml.h / ggml-impl.h / ggml-cuda.h
# live alongside the vendored common.cuh.
MMQ_INCLUDES := -Icuda/mmq
# -lcuda is required for the in-process VMM weight arena (CUDA driver API).
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcuda
MMQ_OBJS := cuda/mmq/ds4_ggml_stubs.o cuda/mmq/ds4_mmq.o cuda/mmq/ds4_mmq_d2r.o cuda/mmq/quantize.o cuda/mmq/mmid.o cuda/mmq/mmvq.o cuda/mmq/ds4_repack.o
CORE_OBJS = ds4.o ds4_image.o ds4_distributed.o ds4_cuda.o $(MMQ_OBJS)
CPU_CORE_OBJS = ds4_cpu.o ds4_image.o ds4_distributed.o
METAL_LDLIBS := $(LDLIBS)
CUDA_EXTRA_BINS := ds4_weight_server
endif

.PHONY: all help clean test cpu cuda cuda-spark cuda-generic cuda-regression \
        proof-cuda-smoke proof-cuda-long proof-cuda-opp-c print-version

ifeq ($(UNAME_S),Darwin)
all: ds4 ds4-server ds4-bench ds4-eval ds4-agent

help:
	@echo "DS4 build targets:"
	@echo "  make              Build Metal ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make cpu          Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make test         Build and run tests"
	@echo "  make clean        Remove build outputs"

ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: ds4_server.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_server.o ds4_kvstore.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-bench: ds4_bench.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_bench.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-eval: ds4_eval.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_eval.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-agent: ds4_agent.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_agent.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_agent_cpu.o ds4_web.o ds4_kvstore.o linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_kvstore.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_web.o ds4_kvstore.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression:
	@echo "cuda-regression requires a CUDA build"

proof-cuda-smoke proof-cuda-long proof-cuda-opp-c:
	@echo "$@ requires a CUDA build"
else
all: help

help:
	@echo "DS4 build targets:"
	@echo "  make cuda-spark          Build CUDA for DGX Spark / GB10 (sm_121a arch alias)"
	@echo "  make cuda-generic        Build CUDA for a generic local CUDA GPU"
	@echo "  make cuda CUDA_ARCH=sm_N Build CUDA with an explicit nvcc -arch value"
	@echo "  make cpu                 Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make test                Build and run tests (reuses the last cuda-* configuration)"
	@echo "  make clean               Remove build outputs (keeps the recorded cuda configuration)"

# GB10 / DGX Spark is compute capability 12.1. Without an explicit -arch,
# nvcc 13.0 emits compute_75 PTX that the driver JITs onto sm_121 with
# Turing-era codepaths (no cp.async, no Blackwell MMA) — measurably slower
# MMQ prefill. The arch must reach the sub-make as CUDA_ARCH (not inside a
# pre-expanded NVCCFLAGS, where the parent's empty NVCC_ARCH_FLAGS would
# erase it), so spark defines travel via NVCC_EXTRA_FLAGS instead.
cuda-spark:
	@printf '%s\n' '# written by make cuda-spark (see the config include note in Makefile)' 'CUDA_ARCH := sm_121' 'NVCC_EXTRA_FLAGS :=' > .ds4-cuda-config.mk
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent $(CUDA_EXTRA_BINS) CUDA_ARCH=sm_121 NVCC_EXTRA_FLAGS=""

cuda-generic:
	@printf '%s\n' '# written by make cuda-generic (see the config include note in Makefile)' 'CUDA_ARCH := native' > .ds4-cuda-config.mk
	$(MAKE) ds4 ds4-server ds4-bench ds4-eval ds4-agent $(CUDA_EXTRA_BINS) CUDA_ARCH=native

cuda:
	@if [ -z "$(strip $(CUDA_ARCH))" ]; then \
		echo "error: specify CUDA_ARCH, for example: make cuda CUDA_ARCH=sm_120"; \
		echo "       or use make cuda-spark / make cuda-generic"; \
		exit 2; \
	fi
	@printf '%s\n' '# written by make cuda (see the config include note in Makefile)' 'CUDA_ARCH := $(strip $(CUDA_ARCH))' > .ds4-cuda-config.mk
	$(MAKE) ds4 ds4-server ds4-bench ds4-eval ds4-agent $(CUDA_EXTRA_BINS) CUDA_ARCH="$(CUDA_ARCH)"

ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-server: ds4_server.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-bench: ds4_bench.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-eval: ds4_eval.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-agent: ds4_agent.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_agent_cpu.o ds4_web.o ds4_kvstore.o linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_kvstore.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_web.o ds4_kvstore.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression: tests/cuda_long_context_smoke
	./tests/cuda_long_context_smoke

# Proof-harness scenarios. Each is a thin wrapper around tests/ds4_proof.py
# --scenario <name>. They expect DS4_PROOF_BASE (and, for MTP scenarios,
# DS4_PROOF_MTP) in the environment; ds4 must already be built. The harness
# materializes the (canonical x overlay) matrix, writes work_dir/expanded-plan.json,
# runs every cell, and reports per-cell selected-token-id MD5s with vs-canonical-
# counterpart parity contracts.
#   - smoke / long: capture-vs-eager PARITY (two paths in one build must match).
#   - opp-c: FP8 KV DRIFT gate. Single canonical, no parity contract; each cell
#     is checked against a committed golden snapshot (tests/proof/expected/
#     cuda-opp-c-full.json) so lossy-FP8 numeric drift between builds is caught.
#     Regenerate the golden after an intentional output change with:
#       tests/ds4_proof.py --scenario cuda-opp-c-full \
#         --write-expected tests/proof/expected/cuda-opp-c-full.json [weight-server flags]
DS4_PROOF_REQUIRE_BASE := @if [ -z "$$DS4_PROOF_BASE" ]; then echo "$@: set DS4_PROOF_BASE to a base model gguf path" >&2; exit 2; fi

proof-cuda-smoke: ds4
	$(DS4_PROOF_REQUIRE_BASE)
	tests/ds4_proof.py --scenario cuda-capture-smoke --work-dir /tmp/ds4_proof/$@

proof-cuda-long: ds4
	$(DS4_PROOF_REQUIRE_BASE)
	tests/ds4_proof.py --scenario cuda-long-context-full --work-dir /tmp/ds4_proof/$@

proof-cuda-opp-c: ds4
	$(DS4_PROOF_REQUIRE_BASE)
	tests/ds4_proof.py --scenario cuda-opp-c-full --work-dir /tmp/ds4_proof/$@ \
		--check-expected tests/proof/expected/cuda-opp-c-full.json
endif

ds4.o: ds4.c ds4.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h ds4_distributed.h ds4_gpu.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_cli.o: ds4_cli.c ds4.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h ds4_distributed.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_distributed.o: ds4_distributed.c ds4_distributed.h ds4.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h
	$(CC) $(CFLAGS) -c -o $@ ds4_distributed.c

# Version stamp: git describe in a checkout; the committed VERSION file
# (bumped at each release cut) covers gitless trees and tag-less clones.
# No --always in the describe tier: on an installer clone without tags it
# "succeeds" with a bare hash, VERSION is never read, and the update check
# treats the unparseable local as older -> daily self-nag on release builds.
DS4_BUILD_VERSION := $(shell git describe --tags --dirty 2>/dev/null || cat VERSION 2>/dev/null || echo unknown)

print-version:
	@echo $(DS4_BUILD_VERSION)

ds4_server.o: ds4_server.c ds4.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h ds4_distributed.h ds4_kvstore.h rax.h Makefile VERSION
	$(CC) $(CFLAGS) -DDS4_BUILD_VERSION='"$(DS4_BUILD_VERSION)"' -c -o $@ ds4_server.c

ds4_bench.o: ds4_bench.c ds4.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h
	$(CC) $(CFLAGS) -c -o $@ ds4_bench.c

ds4_eval.o: ds4_eval.c ds4.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h
	$(CC) $(CFLAGS) -c -o $@ ds4_eval.c

ds4_agent.o: ds4_agent.c ds4.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h ds4_distributed.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_agent.c

ds4_web.o: ds4_web.c ds4_web.h
	$(CC) $(CFLAGS) -c -o $@ ds4_web.c

ds4_kvstore.o: ds4_kvstore.c ds4_kvstore.h ds4.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h
	$(CC) $(CFLAGS) -c -o $@ ds4_kvstore.c

# Track B inc 0: image substrate (upstream antirez/ds4 110afdd, DeepSeek half;
# libc + the vendored iris decoders only, no engine symbols).
ds4_image.o: ds4_image.c ds4_image.h third_party/iris/jpeg.h third_party/iris/png.h
	$(CC) $(CFLAGS) -c -o $@ ds4_image.c

tests/test_deepseek4_vision_image.o: tests/test_deepseek4_vision_image.c ds4_image.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_deepseek4_vision_image.c

tests/test_deepseek4_vision_image: tests/test_deepseek4_vision_image.o ds4_image.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

# Track B inc 2: the encoder oracle harness (built unchanged against upstream
# antirez/ds4 @110afdd; the two dumps must be byte-identical).
tests/dump_vision_embedding.o: tests/dump_vision_embedding.c ds4.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/dump_vision_embedding.c

tests/dump_vision_embedding: tests/dump_vision_embedding.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ tests/dump_vision_embedding.o $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(NVCC) $(NVCCFLAGS) -o $@ tests/dump_vision_embedding.o $(CORE_OBJS) $(CUDA_LDLIBS)
endif

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h ds4_distributed.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_long_context_smoke.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_cpu.o: ds4.c ds4.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h ds4_distributed.h ds4_gpu.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4.c

ds4_cli_cpu.o: ds4_cli.c ds4.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h ds4_distributed.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_cli.c

ds4_server_cpu.o: ds4_server.c ds4.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h ds4_distributed.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_server.c

ds4_bench_cpu.o: ds4_bench.c ds4.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_bench.c

ds4_eval_cpu.o: ds4_eval.c ds4.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_eval.c

ds4_agent_cpu.o: ds4_agent.c ds4.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h ds4_distributed.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_agent.c

ds4_metal.o: ds4_metal.m ds4_gpu.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m

ds4_cuda.o: ds4_cuda.cu ds4_gpu.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h ds4_iq2_tables_cuda.inc cuda/mmq/ds4_repack.h ds4_vit_common.cuh ds4_deepseek4_vision_gpu.cuh
	$(NVCC) $(NVCCFLAGS) -c -o $@ ds4_cuda.cu

# Vendored mmq pieces. ds4_mmq.cu transitively pulls in mmq.cuh which has
# heavy template instantiation - compile in its own TU and link in.
cuda/mmq/ds4_ggml_stubs.o: cuda/mmq/ds4_ggml_stubs.cu cuda/mmq/ds4_ggml_stubs.h cuda/mmq/common.cuh
	$(NVCC) $(NVCCFLAGS) $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/ds4_mmq.o: cuda/mmq/ds4_mmq.cu cuda/mmq/ds4_mmq.h cuda/mmq/ds4_mmq_d2r.cuh cuda/mmq/mmq.cuh cuda/mmq/common.cuh cuda/mmq/quantize.cuh cuda/mmq/mmid.cuh cuda/mmq/vecdotq.cuh cuda/mmq/mma.cuh
	$(NVCC) $(NVCCFLAGS) $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/ds4_mmq_d2r.o: cuda/mmq/ds4_mmq_d2r.cu cuda/mmq/ds4_mmq_d2r.cuh cuda/mmq/mmq.cuh cuda/mmq/common.cuh cuda/mmq/vecdotq.cuh cuda/mmq/mma.cuh
	$(NVCC) $(NVCCFLAGS) $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/quantize.o: cuda/mmq/quantize.cu cuda/mmq/quantize.cuh cuda/mmq/common.cuh cuda/mmq/mmq.cuh
	$(NVCC) $(NVCCFLAGS) $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/mmid.o: cuda/mmq/mmid.cu cuda/mmq/mmid.cuh cuda/mmq/common.cuh
	$(NVCC) $(NVCCFLAGS) $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/mmvq.o: cuda/mmq/mmvq.cu cuda/mmq/mmvq.cuh cuda/mmq/common.cuh cuda/mmq/quantize.cuh cuda/mmq/vecdotq.cuh cuda/mmq/unary.cuh
	$(NVCC) $(NVCCFLAGS) $(MMQ_INCLUDES) -c -o $@ $<

# Shared aligned-artifact layout library: compiled into the engine (via
# MMQ_OBJS) and linked into ds4_weight_server so both producers build
# bit-identical repack artifacts.
cuda/mmq/ds4_repack.o: cuda/mmq/ds4_repack.cu cuda/mmq/ds4_repack.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o ds4.o ds4_distributed.o ds4_cuda.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4_weight_server: tools/ds4_weight_server.cu cuda/mmq/ds4_repack.o
	$(NVCC) $(NVCCFLAGS) -o $@ tools/ds4_weight_server.cu cuda/mmq/ds4_repack.o $(CUDA_LDLIBS)

ds4_test: ds4_test.o ds4_kvstore.o rax.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ ds4_test.o ds4_kvstore.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(NVCC) $(NVCCFLAGS) -o $@ ds4_test.o ds4_kvstore.o rax.o $(CORE_OBJS) $(CUDA_LDLIBS)
endif

test: ds4_test ds4-eval tests/test_deepseek4_vision_image
	./ds4-eval --self-test-extractors
	./ds4_test
	./tests/test_deepseek4_vision_image

clean:
	rm -f ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4_weight_server ds4_cpu ds4_native ds4_server_test ds4_test *.o tests/cuda_long_context_smoke tests/cuda_long_context_smoke.o tests/test_deepseek4_vision_image tests/test_deepseek4_vision_image.o tests/dump_vision_embedding tests/dump_vision_embedding.o
