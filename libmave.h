#include <assert.h>
#include <xmmintrin.h>

/**
 * Helper for defining the alignment of a variable.
 */
#ifndef MV_ALIGNMENT
#define MV_ALIGNMENT(X) __attribute((aligned(X)))
#endif

/**
 * Alignment based on AVX/SIMD support.
 */
#ifdef __AVX__
#define MV_SIMD_ALIGN MV_ALIGNMENT(32)
#else
#define MV_SIMD_ALIGN MV_ALIGNMENT(16)
#endif

// Vectors
typedef float vec2[2];
typedef float vec3[3];
typedef MV_SIMD_ALIGN float vec4[4];

// Matrices
typedef MV_SIMD_ALIGN vec2 m2x2[2];
typedef vec3 m2x3[2];
typedef vec4 m2x4[2];

typedef vec2 m3x2[3];
typedef vec3 m3x3[3];
typedef vec4 m3x4[3];

typedef vec2 m4x2[4];
typedef vec3 m4x3[4];
typedef MV_SIMD_ALIGN vec4 m4x4[4];

// vec2 defs
void vec2_add(vec2 a, vec2 b, vec2 result);
void vec2_sub(vec2 a, vec2 b, vec2 result);
void vec2_mul(vec2 a, vec2 b, vec2 result);
void vec2_div(vec2 a, vec2 b, vec2 result);

void vec3_add(vec3 a, vec3 b, vec3 result);
void vec3_sub(vec3 a, vec3 b, vec3 result);
void vec3_mul(vec3 a, vec3 b, vec3 result);
void vec3_div(vec3 a, vec3 b, vec3 result);

void vec4_add(vec4 a, vec4 b, vec4 result);
void vec4_sub(vec4 a, vec4 b, vec4 result);
void vec4_mul(vec4 a, vec4 b, vec4 result);
void vec4_div(vec4 a, vec4 b, vec4 result);
