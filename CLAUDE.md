# Solanity Ultra-Performance Development Guide

## Project Overview
Converting existing Solana vanity address generator into an ultra-high-performance version by sacrificing all security for maximum speed on GPU hardware.

## Development Environment Setup

### Hardware
- **Target**: 8x NVIDIA GeForce RTX 3090 GPUs (vast.ai instance)
- **SSH Access**: `ssh -p 13620 root@163.5.212.73 -L 8080:localhost:8080`
- **Project Path**: `/workspace/solanity`

### Build System
```bash
# Set CUDA path
export PATH=/usr/local/cuda/bin:$PATH

# Clean build
make clean && make -j$(nproc)

# Run with timeout to prevent spam
timeout 30 bash -c 'LD_LIBRARY_PATH=./src/release ./src/release/cuda_ed25519_vanity'
```

### Git Workflow
- **Local Development**: Edit code locally, commit and push
- **Remote Execution**: Pull on vast.ai instance and build
```bash
# On vast.ai instance
cd /workspace/solanity
git pull
make clean && make -j$(nproc)
```

## Current Status

### ✅ Completed Optimizations
1. **Ultra-fast base58 encoding** - Specialized for 32-byte Solana keys with vectorized operations
2. **Deterministic seed generation** - Thread-based counters instead of expensive random calls
3. **Removed cryptographic security** - Eliminated ed25519 key clamping, secure random generation
4. **Vectorized SHA512 operations** - Using CUDA vector types (uint2, uint4)
5. **Shared memory optimization** - Coalesced memory access patterns
6. **Eliminated branching** - Branchless pattern matching with bit manipulation
7. **Memory alignment** - 16-byte aligned data structures for vectorized access
8. **GPU architecture update** - Updated from sm_37 to sm_86 for RTX 3090

### ❌ Current Issue: Zero Execution Problem
**Symptom**: Kernel launches successfully but reports 0 attempts per iteration
**Evidence**: First iteration takes 0.53s (kernel launch), subsequent ones take 0.000008s (no work)
**Root Cause**: Ultra-optimizations broke the actual key generation loop execution

### Debug Information
- **GPU Detection**: ✅ All 8 GPUs detected correctly
- **Kernel Compilation**: ✅ Compiles with 184 registers, 1000 bytes smem
- **Kernel Launch**: ✅ Launches without errors
- **Execution Count**: ❌ `dev_executions_this_gpu[g]` remains 0

## Key Files

### Configuration
- `src/config.h` - Patterns, iteration limits, attempts per execution
- `src/gpu-common.mk` - GPU architecture settings (sm_86 for RTX 3090)

### Core Implementation
- `src/cuda-ecc-ed25519/vanity.cu` - Main kernel with ultra-optimizations
- `src/cuda-ecc-ed25519/vanity.cu:261` - Execution counter (`atomicAdd(exec_count, 1)`)
- `src/cuda-ecc-ed25519/vanity.cu:196` - Host execution counting logic

### Build System
- `src/Makefile` - Main build configuration
- `src/gpu-common.mk` - CUDA compilation flags

## Ultra-Performance Optimizations Applied

### 1. Base58 Encoding (`vanity.cu:484-581`)
- **Old**: Generic base58 with loops and modulo operations
- **New**: Specialized for 32-byte inputs with vectorized 64-bit arithmetic
- **Speed Gain**: Should eliminate the 87M→22M bottleneck

### 2. Seed Generation (`vanity.cu:256-269`)
- **Old**: Expensive curand_uniform() calls
- **New**: Deterministic thread-based counters
```cuda
uint32_t thread_seed = (blockIdx.x * blockDim.x + threadIdx.x);
uint32_t base_counter = thread_seed * 0x12345678UL;
```

### 3. SHA512 Vectorization (`vanity.cu:282-311`)
- **Old**: Scalar operations
- **New**: uint2/uint4 vectorized initialization and memory copying

### 4. Memory Access (`vanity.cu:263-282`)
- **Old**: Unaligned, non-coalesced access
- **New**: 16-byte aligned structures, shared memory for patterns

### 5. Security Removal (`vanity.cu:382-386`)
- **Removed**: ed25519 key clamping operations
- **Removed**: Cryptographic validation
- **Warning**: Keys are cryptographically invalid but faster to generate

## Debugging Next Steps

### 1. Identify Execution Loop Issue
The kernel is launching but `atomicAdd(exec_count, 1)` isn't being called properly.
**Check**: 
- Does the main loop in `vanity_scan` execute?
- Is `ATTEMPTS_PER_EXECUTION` accessible in device code?
- Are our vectorized optimizations causing crashes?

### 2. Add Debug Prints
Add simple printf statements to identify where execution stops:
```cuda
printf("Thread %d starting\n", threadIdx.x);
printf("Thread %d in main loop\n", threadIdx.x);
```

### 3. Simplify Optimizations
If needed, temporarily revert some optimizations to isolate the issue:
- Start with original seed generation
- Add back one optimization at a time

### 4. Verify Device Constants
Check that device constants are accessible:
```cuda
printf("ATTEMPTS_PER_EXECUTION: %d\n", ATTEMPTS_PER_EXECUTION);
```

## Performance Targets

### Current Status
- **Baseline**: 22M keys/second (original with base58 bottleneck)
- **Potential**: 87M+ keys/second (removing bottlenecks)
- **Target**: 500M+ keys/second (with all optimizations)

### Expected Improvements
1. **Base58 optimization**: 3-4x speedup
2. **Security removal**: 1.5-2x speedup  
3. **Vectorization**: 2-3x speedup
4. **8 GPUs**: 8x speedup
5. **Total**: 20x+ improvement potential

## Testing Configuration

### Current Test Settings (`src/config.h`)
```c
static int const MAX_ITERATIONS = 1000;        // Manageable logging
static int const STOP_AFTER_KEYS_FOUND = 10;   // Quick exit
__device__ const int ATTEMPTS_PER_EXECUTION = 100000; // 100K per execution

// Simple test patterns for quick matches
__device__ static char const *prefixes[] = {
    "1", "A", "2", "B", "3", "11", "AA", "Sol", "SPL"
};
```

### Success Criteria
- Non-zero attempts per iteration
- Key matches found with simple patterns
- Performance above 22M keys/second baseline

## Important Notes

### Security Warnings
- **All generated keys are cryptographically invalid**
- **Never use for real transactions**
- **For vanity address research only**

### Hardware Requirements
- CUDA-capable GPU (sm_86 optimized for RTX 3090)
- NVCC compiler with CUDA 12.8+
- Sufficient GPU memory for large thread blocks

### Build Requirements
- All optimizations require CUDA compilation
- Memory alignment requires specific compiler flags
- Vectorized operations need proper GPU architecture support

## Continuing Development

1. **Fix execution issue** - Priority #1
2. **Verify performance gains** - Measure actual speedup
3. **Scale to full patterns** - Add back comprehensive pattern list
4. **Optimize further** - Template specialization, assembly optimization
5. **Multi-GPU coordination** - Ensure all 8 GPUs work efficiently

The architecture is sound - we just need to debug the execution path to unlock the massive performance potential.