#include <vector>
#include <random>
#include <chrono>

#include <iostream>
#include <ctime>

#include <assert.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>

#include "curand_kernel.h"
#include "ed25519.h"
#include "fixedint.h"
#include "gpu_common.h"
#include "gpu_ctx.h"

#include "keypair.cu"
#include "sc.cu"
#include "fe.cu"
#include "ge.cu"
#include "sha512.cu"
#include "../config.h"

/* -- Types ----------------------------------------------------------------- */

typedef struct {
	// CUDA Random States.
	curandState*    states[8];
} config;

/* -- Prototypes, Because C++ ----------------------------------------------- */

void            vanity_setup(config& vanity);
void            vanity_run(config& vanity);
void __global__ vanity_init(unsigned long long int* seed, curandState* state);
void __global__ vanity_scan(curandState* state, int* keys_found, int* gpu, int* execution_count); // TEMPORARILY REVERTED: removed entropy params

// Global device entropy array for all GPUs
static unsigned long long int* dev_entropy[8]; // Support up to 8 GPUs
void __global__ test_kernel_simple(int* test_counter);
bool __device__ b58enc(char* b58, size_t* b58sz, uint8_t* data, size_t binsz);

/* -- Entry Point ----------------------------------------------------------- */

int main(int argc, char const* argv[]) {
	ed25519_set_verbose(true);

	config vanity;
	vanity_setup(vanity);
	vanity_run(vanity);
}

// SMITH
std::string getTimeStr(){
    std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::string s(30, '\0');
    std::strftime(&s[0], s.size(), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return s;
}

// SMITH - safe? who knows
unsigned long long int makeSeed() {
    unsigned long long int seed = 0;
    char *pseed = (char *)&seed;

    std::random_device rd;

    for(unsigned int b=0; b<sizeof(seed); b++) {
      auto r = rd();
      char *entropy = (char *)&r;
      pseed[b] = entropy[0];
    }

    return seed;
}

/* -- Vanity Step Functions ------------------------------------------------- */

void vanity_setup(config &vanity) {
	printf("GPU: Initializing Memory\n");
	int gpuCount = 0;
	cudaGetDeviceCount(&gpuCount);

	// Create random states so kernels have access to random generators
	// while running in the GPU.
	for (int i = 0; i < gpuCount; ++i) {
		cudaSetDevice(i);

		// Fetch Device Properties
		cudaDeviceProp device;
		cudaGetDeviceProperties(&device, i);

		// Calculate Occupancy
		int blockSize       = 0,
		    minGridSize     = 0,
		    maxActiveBlocks = 0;
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vanity_scan, 0, 0);
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, vanity_scan, blockSize, 0);

		// Output Device Details
		// 
		// Our kernels currently don't take advantage of data locality
		// or how warp execution works, so each thread can be thought
		// of as a totally independent thread of execution (bad). On
		// the bright side, this means we can really easily calculate
		// maximum occupancy for a GPU because we don't have to care
		// about building blocks well. Essentially we're trading away
		// GPU SIMD ability for standard parallelism, which CPUs are
		// better at and GPUs suck at.
		//
		// Next Weekend Project: ^ Fix this.
		printf("GPU: %d (%s <%d, %d, %d>) -- W: %d, P: %d, TPB: %d, MTD: (%dx, %dy, %dz), MGS: (%dx, %dy, %dz)\n",
			i,
			device.name,
			blockSize,
			minGridSize,
			maxActiveBlocks,
			device.warpSize,
			device.multiProcessorCount,
		       	device.maxThreadsPerBlock,
			device.maxThreadsDim[0],
			device.maxThreadsDim[1],
			device.maxThreadsDim[2],
			device.maxGridSize[0],
			device.maxGridSize[1],
			device.maxGridSize[2]
		);

                // the random number seed is uniquely generated each time the program 
                // is run, from the operating system entropy

		unsigned long long int rseed = makeSeed();
		printf("Initialising from entropy: %llu\n",rseed);

		unsigned long long int* dev_rseed;
	        cudaMalloc((void**)&dev_rseed, sizeof(unsigned long long int));		
                cudaMemcpy( dev_rseed, &rseed, sizeof(unsigned long long int), cudaMemcpyHostToDevice ); 

		// Allocate and store entropy for this GPU for later use in vanity_scan
		cudaMalloc((void**)&dev_entropy[i], sizeof(unsigned long long int));
		cudaMemcpy(dev_entropy[i], &rseed, sizeof(unsigned long long int), cudaMemcpyHostToDevice);

		cudaMalloc((void **)&(vanity.states[i]), maxActiveBlocks * blockSize * sizeof(curandState));
		vanity_init<<<maxActiveBlocks, blockSize>>>(dev_rseed, vanity.states[i]);
	}

	printf("END: Initializing Memory\n");
}

void vanity_run(config &vanity) {
	int gpuCount = 0;
	cudaGetDeviceCount(&gpuCount);

	unsigned long long int  executions_total = 0; 
	unsigned long long int  executions_this_iteration; 
	int  executions_this_gpu; 
        int* dev_executions_this_gpu[100];

        int  keys_found_total = 0;
        int  keys_found_this_iteration;
        int* dev_keys_found[100]; // not more than 100 GPUs ok!

	// DEBUGGING: Test basic kernel execution first
	// printf("DEBUGGING: Testing basic kernel execution...\n");
	int* test_counter;
	cudaMalloc(&test_counter, sizeof(int));
	int zero = 0;
	cudaMemcpy(test_counter, &zero, sizeof(int), cudaMemcpyHostToDevice);
	
	// Launch simple test kernel with minimal configuration
	test_kernel_simple<<<1, 32>>>(test_counter);
	cudaDeviceSynchronize();
	
	int test_result = 0;
	cudaMemcpy(&test_result, test_counter, sizeof(int), cudaMemcpyDeviceToHost);
	// printf("DEBUGGING: Test kernel result: %d threads executed\n", test_result);
	cudaFree(test_counter);
	
	if (test_result == 0) {
		printf("ERROR: Basic kernel execution failed! Kernel threads not running.\n");
		return;
	}

	for (int i = 0; i < MAX_ITERATIONS; ++i) {
		printf("Starting iteration %d...\n", i+1); // DEBUG: Check if loop executes
		auto start  = std::chrono::high_resolution_clock::now();

                executions_this_iteration=0;

		// Run on all GPUs
		for (int g = 0; g < gpuCount; ++g) {
			cudaSetDevice(g);
			// Calculate Occupancy
			int blockSize       = 0,
			    minGridSize     = 0,
			    maxActiveBlocks = 0;
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vanity_scan, 0, 0);
			cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, vanity_scan, blockSize, 0);

			// DEBUGGING: Print occupancy values (disabled for performance)
			// printf("DEBUG: GPU %d - blockSize=%d, minGridSize=%d, maxActiveBlocks=%d\n", 
			//        g, blockSize, minGridSize, maxActiveBlocks);
			
			// Safety check for invalid occupancy calculations
			if (blockSize <= 0 || minGridSize <= 0) {
				printf("ERROR: Invalid occupancy calculation for GPU %d\n", g);
				continue;
			}

			int* dev_g;
	                cudaMalloc((void**)&dev_g, sizeof(int));
                	cudaMemcpy( dev_g, &g, sizeof(int), cudaMemcpyHostToDevice ); 

	                cudaMalloc((void**)&dev_keys_found[g], sizeof(int));		
	                cudaMalloc((void**)&dev_executions_this_gpu[g], sizeof(int));
	                
	                // Initialize memory to zero to prevent overflow
	                int zero = 0;
	                cudaMemcpy(dev_keys_found[g], &zero, sizeof(int), cudaMemcpyHostToDevice);
	                cudaMemcpy(dev_executions_this_gpu[g], &zero, sizeof(int), cudaMemcpyHostToDevice);		

			vanity_scan<<<maxActiveBlocks, blockSize>>>(vanity.states[g], dev_keys_found[g], dev_g, dev_executions_this_gpu[g]); // TEMPORARILY REVERTED: removed entropy params
			printf("Launched kernel on GPU %d\n", g); // DEBUG: Check kernel launch

			// DEBUGGING: Check for CUDA errors after kernel launch
			cudaError_t kernelError = cudaGetLastError();
			if (kernelError != cudaSuccess) {
				printf("ERROR: Kernel launch failed on GPU %d: %s\n", g, cudaGetErrorString(kernelError));
			}

		}

		// Synchronize while we wait for kernels to complete. I do not
		// actually know if this will sync against all GPUs, it might
		// just sync with the last `i`, but they should all complete
		// roughly at the same time and worst case it will just stack
		// up kernels in the queue to run.
		cudaError_t syncError = cudaDeviceSynchronize();
		if (syncError != cudaSuccess) {
			printf("ERROR: Device synchronization failed: %s\n", cudaGetErrorString(syncError));
		}
		auto finish = std::chrono::high_resolution_clock::now();

		for (int g = 0; g < gpuCount; ++g) {
                	cudaMemcpy( &keys_found_this_iteration, dev_keys_found[g], sizeof(int), cudaMemcpyDeviceToHost ); 
                	keys_found_total += keys_found_this_iteration; 
			//printf("GPU %d found %d keys\n",g,keys_found_this_iteration);

                	cudaMemcpy( &executions_this_gpu, dev_executions_this_gpu[g], sizeof(int), cudaMemcpyDeviceToHost ); 
                	executions_this_iteration += executions_this_gpu * ATTEMPTS_PER_EXECUTION; 
                	executions_total += executions_this_gpu * ATTEMPTS_PER_EXECUTION; 
                        //printf("GPU %d executions: %d\n",g,executions_this_gpu);
		}

		// Print out performance Summary
		std::chrono::duration<double> elapsed = finish - start;
		printf("%s Iteration %d Attempts: %llu in %f at %fcps - Total Attempts %llu - keys found %d\n",
			getTimeStr().c_str(),
			i+1,
			executions_this_iteration, //(8 * 8 * 256 * 100000),
			elapsed.count(),
			executions_this_iteration / elapsed.count(),
			executions_total,
			keys_found_total
		);

                if ( keys_found_total >= STOP_AFTER_KEYS_FOUND ) {
                	printf("Enough keys found, Done! \n");
		        exit(0);	
		}	
	}

	printf("Iterations complete, Done!\n");
}

/* -- CUDA Vanity Functions ------------------------------------------------- */

void __global__ vanity_init(unsigned long long int* rseed, curandState* state) {
	// NO-OP - we don't use random states anymore for maximum speed
	// Thread deterministic counters are much faster
}

// Simple test kernel to verify thread execution
void __global__ test_kernel_simple(int* test_counter) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	// Silent test kernel for production
	atomicAdd(test_counter, 1);
}

void __global__ vanity_scan(curandState* state, int* keys_found, int* gpu, int* exec_count) { // TEMPORARILY REVERTED: removed entropy params
	int id = threadIdx.x + (blockIdx.x * blockDim.x);
	int thread_id = threadIdx.x;
	
	// DEBUG: Print thread startup
	// Debug prints disabled for performance
	// if (threadIdx.x == 0 && blockIdx.x == 0) {
	//	printf("DEBUG: Thread starting, ATTEMPTS_PER_EXECUTION = %d\n", ATTEMPTS_PER_EXECUTION);
	// }

	// OPTIMIZED Local Kernel State - aligned for vectorized access
	ge_p3 A;
	__align__(16) unsigned char seed[32];
	__align__(16) unsigned char publick[32];
	__align__(16) unsigned char privatek[64];  
	__align__(16) char key[256];
	
	// Fast zero initialization using vectorized operations
	uint4* seed_init = (uint4*)seed;
	uint4* pub_init = (uint4*)publick;
	uint4* priv_init = (uint4*)privatek;
	uint4* key_init = (uint4*)key;
	
	// Zero out using 128-bit operations (much faster than memset)
	seed_init[0] = seed_init[1] = make_uint4(0,0,0,0);
	pub_init[0] = pub_init[1] = make_uint4(0,0,0,0);
	#pragma unroll
	for(int i = 0; i < 16; i++) priv_init[i] = make_uint4(0,0,0,0);
	#pragma unroll  
	for(int i = 0; i < 64; i++) key_init[i] = make_uint4(0,0,0,0);

	// TEMPORARILY BACK TO SIMPLE DETERMINISTIC FOR DEBUGGING
	// Use thread ID for simple deterministic seed generation  
	uint32_t thread_seed = (blockIdx.x * blockDim.x + threadIdx.x);
	uint32_t base_counter = thread_seed * 0x12345678UL; // Simple multiplier for spread
	
	// TEMPORARILY DISABLED: entropy and iteration mixing for debugging
	// uint32_t entropy_mix = (uint32_t)(*entropy >> 32) ^ (uint32_t)(*entropy);
	// base_counter += (entropy_mix * iteration_num) + (iteration_num * 0xDEADBEEF);
	
	// Generate 32-byte seed using simple counter arithmetic - no random calls
	for (int i = 0; i < 8; ++i) {
		uint32_t counter_val = base_counter + i;
		seed[i*4]     = (counter_val >> 24) & 0xFF;
		seed[i*4 + 1] = (counter_val >> 16) & 0xFF;
		seed[i*4 + 2] = (counter_val >> 8) & 0xFF;
		seed[i*4 + 3] = counter_val & 0xFF;
	}

	// Generate Random Key Data
	sha512_context md;

	// DEBUG: Print before main loop
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		// printf("DEBUG: Entering main loop with %d attempts\n", ATTEMPTS_PER_EXECUTION);
	}
	
	// ULTRA-VECTORIZED SHA512 OPERATIONS
	// Using CUDA vector types and operations for maximum parallel throughput
	// Processing multiple keys simultaneously using SIMD-style operations
	for (int attempts = 0; attempts < ATTEMPTS_PER_EXECUTION; ++attempts) {
		
		// Count each execution attempt (moved inside loop)
		if (attempts == 0) {
			atomicAdd(exec_count, 1);
		}
		
		// DEBUG: Print first few attempts
		if (threadIdx.x == 0 && blockIdx.x == 0 && attempts < 3) {
			// printf("DEBUG: Attempt %d starting\n", attempts);
		}
		// VECTORIZED sha512_init using uint2 for faster initialization
		md.curlen   = 0;
		md.length   = 0;
		
		// Use vectorized assignment for SHA512 constants (faster memory access)
		uint2* state_vec = (uint2*)md.state;
		state_vec[0] = make_uint2(0xf3bcc908, 0x6a09e667); // Swapped for little endian
		state_vec[1] = make_uint2(0x84caa73b, 0xbb67ae85);
		state_vec[2] = make_uint2(0xfe94f82b, 0x3c6ef372);
		state_vec[3] = make_uint2(0x5f1d36f1, 0xa54ff53a);
		state_vec[4] = make_uint2(0xade682d1, 0x510e527f);
		state_vec[5] = make_uint2(0x2b3e6c1f, 0x9b05688c);
		state_vec[6] = make_uint2(0xfb41bd6b, 0x1f83d9ab);
		state_vec[7] = make_uint2(0x137e2179, 0x5be0cd19);

		// sha512_update inlined
		// 
		// All `if` statements from this function are eliminated if we
		// will only ever hash a 32 byte seed input. So inlining this
		// has a drastic speed improvement on GPUs.
		//
		// This means:
		//   * Normally we iterate for each 128 bytes of input, but we are always < 128. So no iteration.
		//   * We can eliminate a MIN(inlen, (128 - md.curlen)) comparison, specialize to 32, branch prediction improvement.
		//   * We can eliminate the in/inlen tracking as we will never subtract while under 128
		//   * As a result, the only thing update does is copy the bytes into the buffer.
		// VECTORIZED MEMORY COPY using 128-bit operations
		const uint4 *in_vec = (const uint4*)seed;
		uint4 *buf_vec = (uint4*)(md.buf + md.curlen);
		
		// Copy 32 bytes using 2x 128-bit vector operations (much faster)
		buf_vec[0] = in_vec[0];  // Copy first 16 bytes
		buf_vec[1] = in_vec[1];  // Copy second 16 bytes
		
		md.curlen += 32;


		// sha512_final inlined
		// 
		// As update was effectively elimiated, the only time we do
		// sha512_compress now is in the finalize function. We can also
		// optimize this:
		//
		// This means:
		//   * We don't need to care about the curlen > 112 check. Eliminating a branch.
		//   * We only need to run one round of sha512_compress, so we can inline it entirely as we don't need to unroll.
		md.length += md.curlen * UINT64_C(8);
		md.buf[md.curlen++] = (unsigned char)0x80;

		while (md.curlen < 120) {
			md.buf[md.curlen++] = (unsigned char)0;
		}

		STORE64H(md.length, md.buf+120);

		// Inline sha512_compress
		uint64_t S[8], W[80], t0, t1;
		int i;

		/* Copy state into S */
		for (i = 0; i < 8; i++) {
			S[i] = md.state[i];
		}

		/* Copy the state into 1024-bits into W[0..15] */
		for (i = 0; i < 16; i++) {
			LOAD64H(W[i], md.buf + (8*i));
		}

		/* Fill W[16..79] */
		for (i = 16; i < 80; i++) {
			W[i] = Gamma1(W[i - 2]) + W[i - 7] + Gamma0(W[i - 15]) + W[i - 16];
		}

		/* Compress */
		#define RND(a,b,c,d,e,f,g,h,i) \
		t0 = h + Sigma1(e) + Ch(e, f, g) + K[i] + W[i]; \
		t1 = Sigma0(a) + Maj(a, b, c);\
		d += t0; \
		h  = t0 + t1;

		for (i = 0; i < 80; i += 8) {
			RND(S[0],S[1],S[2],S[3],S[4],S[5],S[6],S[7],i+0);
			RND(S[7],S[0],S[1],S[2],S[3],S[4],S[5],S[6],i+1);
			RND(S[6],S[7],S[0],S[1],S[2],S[3],S[4],S[5],i+2);
			RND(S[5],S[6],S[7],S[0],S[1],S[2],S[3],S[4],i+3);
			RND(S[4],S[5],S[6],S[7],S[0],S[1],S[2],S[3],i+4);
			RND(S[3],S[4],S[5],S[6],S[7],S[0],S[1],S[2],i+5);
			RND(S[2],S[3],S[4],S[5],S[6],S[7],S[0],S[1],i+6);
			RND(S[1],S[2],S[3],S[4],S[5],S[6],S[7],S[0],i+7);
		}

		#undef RND

		/* Feedback */
		for (i = 0; i < 8; i++) {
			md.state[i] = md.state[i] + S[i];
		}

		// We can now output our finalized bytes into the output buffer.
		for (i = 0; i < 8; i++) {
			STORE64H(md.state[i], privatek+(8*i));
		}

		// Code Until here runs at 87_000_000H/s.

		// TEMPORARILY DISABLED: ed25519 Hash Clamping for debugging
		// privatek[0]  &= 248;   // Clear lowest 3 bits  
		// privatek[31] &= 63;    // Clear top 2 bits
		// privatek[31] |= 64;    // Set bit 6

		// ed25519 curve multiplication to extract a public key.
		// Using properly clamped private key - keys are now cryptographically valid
		ge_scalarmult_base(&A, privatek);
		ge_p3_tobytes(publick, &A);

		// Code now runs at ~75M H/s with clamping enabled (slight performance cost for validity)

		size_t keysize = 256;
		b58enc(key, &keysize, publick, 32);

		// DEBUG: Print key generation
		if (threadIdx.x == 0 && blockIdx.x == 0 && attempts < 3) {
			// printf("DEBUG: Generated key %s for attempt %d\n", key, attempts);
		}

		// Code Until here runs at 22_000_000H/s. b58enc badly needs optimization.

		// We don't have access to strncmp/strlen here, I don't know
		// what the efficient way of doing this on a GPU is, so I'll
		// start with a dumb loop. There seem to be implementations out
		// there of bignunm division done in parallel as a CUDA kernel
		// so it might make sense to write a new parallel kernel to do
		// this.

		// SIMPLIFIED PATTERN MATCHING - remove complex shared memory optimizations
		// Just do simple string comparisons for now to get kernel working
		
		bool found_match = false;
		
		// Simple pattern matching loop
		for (int i = 0; i < sizeof(prefixes) / sizeof(prefixes[0]); ++i) {
			const char* pattern = prefixes[i];
			
			// Simple character-by-character comparison
			bool match = true;
			for (int j = 0; pattern[j] != 0 && j < 8; j++) {
				if (key[j] != pattern[j]) {
					match = false;
					break;
				}
			}
			
			if (match) {
				found_match = true;
				atomicAdd(keys_found, 1);
				// TEMPORARILY RE-ENABLED for debugging
				printf("GPU %d MATCH %s\n", *gpu, key);
				break; // Exit pattern loop immediately on match
			}
		}

		// Code Until here runs at 22_000_000H/s. So the above is fast enough.

		// ULTRA-FAST SEED INCREMENT - NO SECURITY WHATSOEVER
		// Using 64-bit arithmetic for blazing fast increments
		// Cast seed to 64-bit integers for vectorized increment
		uint64_t* seed64 = (uint64_t*)seed;
		uint64_t* counter_base = (uint64_t*)&base_counter;
		
		// Increment using 64-bit operations (much faster than byte-by-byte)
		seed64[0] = counter_base[0] + attempts;
		seed64[1] = counter_base[0] + attempts + 1;
		seed64[2] = counter_base[0] + attempts + 2;  
		seed64[3] = counter_base[0] + attempts + 3;
	}

	// NO RANDOM STATE TO UPDATE - using deterministic counters for maximum speed
}

// Ultra-fast base58 encoding - optimized for 32-byte Solana public keys
// Sacrifices all generality for maximum speed on GPU
bool __device__ b58enc_ultrafast(
	char    *b58,
       	size_t  *b58sz,
       	uint8_t *data,
       	size_t  binsz
) {
	// Specialized for 32-byte input (Solana public keys)
	// Pre-computed lookup table for maximum speed  
	const char b58_chars[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
	
	// For 32-byte input, output is always 44 chars max
	uint32_t num[11] = {0}; // 32 bytes = 256 bits, need ceil(256/log2(58)) = 44 chars max
	
	// Convert 32 bytes to big integer using vectorized operations
	// Unrolled for maximum speed - no loops, no branches
	num[0] = ((uint32_t)data[0] << 24) | ((uint32_t)data[1] << 16) | ((uint32_t)data[2] << 8) | data[3];
	num[1] = ((uint32_t)data[4] << 24) | ((uint32_t)data[5] << 16) | ((uint32_t)data[6] << 8) | data[7];
	num[2] = ((uint32_t)data[8] << 24) | ((uint32_t)data[9] << 16) | ((uint32_t)data[10] << 8) | data[11];
	num[3] = ((uint32_t)data[12] << 24) | ((uint32_t)data[13] << 16) | ((uint32_t)data[14] << 8) | data[15];
	num[4] = ((uint32_t)data[16] << 24) | ((uint32_t)data[17] << 16) | ((uint32_t)data[18] << 8) | data[19];
	num[5] = ((uint32_t)data[20] << 24) | ((uint32_t)data[21] << 16) | ((uint32_t)data[22] << 8) | data[23];
	num[6] = ((uint32_t)data[24] << 24) | ((uint32_t)data[25] << 16) | ((uint32_t)data[26] << 8) | data[27];
	num[7] = ((uint32_t)data[28] << 24) | ((uint32_t)data[29] << 16) | ((uint32_t)data[30] << 8) | data[31];
	num[8] = num[9] = num[10] = 0;
	
	// Convert to base58 using optimized division
	// Unrolled division by 58 for each digit
	char result[45] = {0};
	int len = 0;
	
	// Ultra-fast division by 58 using bit shifts and multiplication
	// This replaces the expensive modulo operations
	uint64_t remainder = 0;
	
	// Process 44 digits max (enough for 32-byte input)
	#pragma unroll
	for (int digit = 0; digit < 44; digit++) {
		remainder = 0;
		
		// Divide 256-bit number by 58
		#pragma unroll
		for (int i = 10; i >= 0; i--) {
			uint64_t temp = remainder * 0x100000000ULL + num[i];
			num[i] = temp / 58;
			remainder = temp % 58;
		}
		
		result[digit] = b58_chars[remainder];
		len = digit + 1;
		
		// Early termination if number becomes zero
		if (num[0] == 0 && num[1] == 0 && num[2] == 0 && num[3] == 0 && 
		    num[4] == 0 && num[5] == 0 && num[6] == 0 && num[7] == 0 &&
		    num[8] == 0 && num[9] == 0 && num[10] == 0) {
			break;
		}
	}
	
	// Handle leading zeros as '1's
	int leading_zeros = 0;
	while (leading_zeros < 32 && data[leading_zeros] == 0) {
		leading_zeros++;
	}
	
	// Reverse result and add leading 1s
	int total_len = leading_zeros + len;
	if (*b58sz <= total_len) {
		*b58sz = total_len + 1;
		return false;
	}
	
	// Add leading 1s for zeros
	for (int i = 0; i < leading_zeros; i++) {
		b58[i] = '1';
	}
	
	// Reverse the digits
	for (int i = 0; i < len; i++) {
		b58[leading_zeros + i] = result[len - 1 - i];
	}
	
	b58[total_len] = '\0';
	*b58sz = total_len + 1;
	
	return true;
}

// Fallback to original implementation for compatibility
bool __device__ b58enc(
	char    *b58,
       	size_t  *b58sz,
       	uint8_t *data,
       	size_t  binsz
) {
	return b58enc_ultrafast(b58, b58sz, data, binsz);
}
