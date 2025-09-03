#ifndef VANITY_CONFIG
#define VANITY_CONFIG

// ULTRA-PERFORMANCE CONFIGURATION FOR SOLANA TOKEN VANITY ADDRESSES
// NO SECURITY - MAXIMUM SPEED CONFIGURATION

static int const MAX_ITERATIONS = 1000;  // Reasonable number for testing
static int const STOP_AFTER_KEYS_FOUND = 100; // Stop after finding more keys for better performance measurement

// Scaled up for performance measurement - better amortization of kernel launch overhead  
__device__ const int ATTEMPTS_PER_EXECUTION = 1000000; // 1M attempts per execution for performance testing

__device__ const int MAX_PATTERNS = 50; // Support many more patterns

// SOLANA TOKEN VANITY PATTERNS - optimized for popular prefixes
// Using ? as wildcard for flexible matching
// Ordered by probability for maximum hit rate

__device__ static char const *prefixes[] = {
    // Simple test patterns - easy to find for debugging
    "1",      // Very common
    "A",      // Very common  
    "2",      // Common
    "B",      // Common
    "3",      // Common
    "11",     // Less common but findable
    "AA",     // Less common but findable
    "Sol",    // Solana
    "SPL",    // SPL Token  
};

#endif
