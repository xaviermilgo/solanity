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

## Testing Instructions

### Remote Development Workflow
1. **Local Development**: Edit code locally, commit and push to git
2. **Remote Testing**: SSH to vast.ai instance, pull changes, build and test

### Remote Machine Access
```bash
# SSH to remote machine with port forwarding
ssh -p 13620 root@163.5.212.73 -L 8080:localhost:8080

# Navigate to project directory
cd /workspace/solanity

# Pull latest changes
git pull

# Build with CUDA
export PATH=/usr/local/cuda/bin:$PATH
make clean && make -j$(nproc)

# Test with timeout to prevent spam
timeout 30 bash -c 'LD_LIBRARY_PATH=./src/release ./src/release/cuda_ed25519_vanity'
```

### Current Debug Status (🔍 ROOT CAUSE IDENTIFIED)

**✅ Identified Issues**:
1. **Complex Kernel Resource Exhaustion**: vanity_scan kernel uses 197 registers + 1000 bytes shared memory
2. **First Iteration Success**: Occupancy calculation works initially (blockSize=256, minGridSize=82)
3. **Subsequent Failures**: Occupancy calculations return 0 after first kernel execution
4. **CUDA Errors**: Kernel execution fails after first iteration, causing invalid launches

**🔧 Debug Fixes Applied**:
1. **CUDA Error Checking**: Added error checking after kernel launch and synchronization
2. **Memory Initialization**: Zero-initialize device memory to prevent overflow readings
3. **Kernel Simplification**: Removed complex shared memory optimizations that exhausted resources
4. **Simple Pattern Matching**: Replaced vectorized pattern matching with basic string comparison

**Evidence Found**:
- ✅ Basic test kernel works (32 threads executed successfully)  
- ✅ First iteration launches (valid occupancy: blockSize=256, minGridSize=82, maxActiveBlocks=1)
- ❌ All subsequent iterations fail (invalid occupancy: blockSize=0, minGridSize=0, maxActiveBlocks=0)
- ❌ Massive overflow numbers indicate uninitialized memory reads before fix

### Expected Debug Output
```
DEBUG: Thread starting, ATTEMPTS_PER_EXECUTION = 100000
DEBUG: Entering main loop with 100000 attempts  
DEBUG: Attempt 0 starting
DEBUG: Generated key [base58_key] for attempt 0
DEBUG: Attempt 1 starting
DEBUG: Generated key [base58_key] for attempt 1
[... execution continues ...]
```

### Next Steps 
1. **Test Simplified Kernel**: Verify execution with reduced resource usage
2. **CUDA Error Analysis**: Check for specific error messages after kernel execution
3. **Performance Baseline**: Measure simplified kernel performance vs original
4. **Gradual Re-optimization**: Add back optimizations one by one once basic execution works

### Technical Root Cause
The ultra-optimizations created a kernel that exceeded RTX 3090 resource limits:
- **197 registers per thread** (likely exceeds optimal occupancy)
- **1000 bytes shared memory** (MAX_PATTERNS * 16 + MAX_PATTERNS * 4 = 50*20 = 1000 bytes)
- **Complex vectorized operations** (uint2/uint4 operations, complex bit manipulation)

When the first kernel execution fails internally, subsequent `cudaOccupancyMaxPotentialBlockSize` calls return 0, preventing further kernel launches. The simplified kernel should use far fewer resources and execute successfully.

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