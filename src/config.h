#ifndef VANITY_CONFIG
#define VANITY_CONFIG

// ULTRA-PERFORMANCE CONFIGURATION FOR SOLANA TOKEN VANITY ADDRESSES
// NO SECURITY - MAXIMUM SPEED CONFIGURATION

static int const MAX_ITERATIONS = 1000000;  // 10x more iterations
static int const STOP_AFTER_KEYS_FOUND = 1000; // Find more keys before stopping

// MASSIVELY INCREASED attempts per execution for maximum throughput  
__device__ const int ATTEMPTS_PER_EXECUTION = 1000000; // 10x increase

__device__ const int MAX_PATTERNS = 50; // Support many more patterns

// SOLANA TOKEN VANITY PATTERNS - optimized for popular prefixes
// Using ? as wildcard for flexible matching
// Ordered by probability for maximum hit rate

__device__ static char const *prefixes[] = {
    // High-probability 3-4 character patterns
    "Sol",    // Solana
    "SPL",    // SPL Token  
    "Moon",   // Popular meme pattern
    "Pump",   // Pump.fun tokens
    "Meme",   // Meme tokens
    "Doge",   // Doge-themed
    "Pepe",   // Pepe tokens
    "Chad",   // Chad tokens
    "Bull",   // Bull market
    "Bear",   // Bear market
    "Ape",    // Ape tokens
    "Wojak",  // Wojak tokens
    "69",     // Popular number
    "420",    // Popular number
    "100",    // Round number
    "1000",   // Round number
    
    // 4-5 character premium patterns  
    "AAAAA",  // Original patterns
    "BBBBB", 
    "CCCCC",
    "11111",
    "22222", 
    "33333",
    "77777",
    "88888",
    "99999",
    "XXXXX",
    "ZZZZZ",
    
    // Token-specific patterns
    "Token",  // Generic token
    "Coin",   // Generic coin
    "Cash",   // Cash tokens
    "Gold",   // Gold-backed
    "Silver", // Silver-backed
    "Rare",   // Rare items
    "Epic",   // Epic items
    "Saga",   // Saga phone integration
    "Magic", // Magic Eden
    "Trade",  // Trading tokens
    "Stake",  // Staking tokens
    "Yield",  // Yield farming
    "Farm",   // Farming tokens
    "Pool",   // Liquidity pools
    "Swap",   // DEX tokens
    "Bridge", // Bridge tokens
    "Cross",  // Cross-chain
    "Layer",  // Layer 2 tokens
    "Fast",   // Fast transactions
    "Cheap",  // Low-cost tokens
    "Green",  // Eco-friendly
};

#endif
