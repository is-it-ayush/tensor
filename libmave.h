#include <assert.h>
#include <immintrin.h>
#include <xmmintrin.h>

#ifndef ALIGNMENT
#define ALIGNMENT(X) __attribute((aligned(X)))
#ifdef __AVX__
#define MV_SIMD_ALIGN ALIGNMENT(32)
#elif __SSE__
#define MV_SIMD_ALIGN ALIGNMENT(16)
#endif
#endif

typedef float vec2[2];
typedef float vec3[3];
typedef float vec4[4];

typedef MV_SIMD_ALIGN vec2 m2x2[2];
typedef MV_SIMD_ALIGN vec3 m2x3[2];
typedef MV_SIMD_ALIGN vec4 m2x4[2];

typedef MV_SIMD_ALIGN vec2 m3x2[3];
typedef MV_SIMD_ALIGN vec3 m3x3[3];
typedef MV_SIMD_ALIGN vec4 m3x4[3];

typedef MV_SIMD_ALIGN vec2 m4x2[4];
typedef MV_SIMD_ALIGN vec3 m4x3[4];
typedef MV_SIMD_ALIGN vec4 m4x4[4];

#define MATH_PI 3.1415926535897932384626433

void vec2_add(vec2 a, vec2 b, vec2 r);
void vec2_sub(vec2 a, vec2 b, vec2 r);
void vec2_dot_mul(vec2 a, vec2 b, vec2 r);
void vec2_div(vec2 a, vec2 b, vec2 r);

void vec3_add(vec3 a, vec3 b, vec3 r);
void vec3_sub(vec3 a, vec3 b, vec3 r);
void vec3_dot_mul(vec3 a, vec3 b, vec3 r);
void vec3_cross_mul(vec3 a, vec3 b, vec3 r);
void vec3_div(vec3 a, vec3 b, vec3 r);

void vec4_add(vec4 a, vec4 b, vec4 r);
void vec4_sub(vec4 a, vec4 b, vec4 r);
void vec4_dot_mul(vec4 a, vec4 b, vec4 r);
void vec4_div(vec4 a, vec4 b, vec4 r);

void m2x2_add(m2x2 a, m2x2 b, m2x2 r);
void m2x2_sub(m2x2 a, m2x2 b, m2x2 r);
void m2x2_mul(m2x2 a, m2x2 b, m2x2 r);
void m2x2_vec2_mul(m2x2 a, vec2 b, vec2 r);

void m2x3_add(m2x3 a, m2x3 b, m2x3 r);
void m2x3_sub(m2x3 a, m2x3 b, m2x3 r);
void m2x3_vec3_mul(m2x3 a, vec3 b, vec2 r);

void m2x4_add(m2x4 a, m2x4 b, m2x4 r);
void m2x4_sub(m2x4 a, m2x4 b, m2x4 r);
void m2x4_vec4_mul(m2x4 a, vec4 b, vec2 r);

void m3x2_add(m3x2 a, m3x2 b, m3x2 r);
void m3x2_sub(m3x2 a, m3x2 b, m3x2 r);
void m3x2_vec2_mul(m3x2 a, vec2 b, vec3 r);

void m3x3_add(m3x3 a, m3x3 b, m3x3 r);
void m3x3_sub(m3x3 a, m3x3 b, m3x3 r);
void m3x3_mul(m3x3 a, m3x3 b, m3x3 r);
void m3x3_vec3_mul(m3x3 a, vec3 b, vec3 r);

void m3x4_add(m3x4 a, m3x4 b, m3x4 r);
void m3x4_sub(m3x4 a, m3x4 b, m3x4 r);
void m3x4_vec4_mul(m3x4 a, vec4 b, vec3 r);

void m4x2_add(m4x2 a, m4x2 b, m4x2 r);
void m4x2_sub(m4x2 a, m4x2 b, m4x2 r);
void m4x2_vec2_mul(m4x2 a, vec2 b, vec4 r);

void m4x3_add(m4x3 a, m4x3 b, m4x3 r);
void m4x3_sub(m4x3 a, m4x3 b, m4x3 r);
void m4x3_vec3_mul(m4x3 a, vec3 b, vec4 r);

void m4x4_add(m4x4 a, m4x4 b, m4x4 r);
void m4x4_sub(m4x4 a, m4x4 b, m4x4 r);
void m4x4_mul(m4x4 a, m4x4 b, m4x4 r);
void m4x4_vec4_mul(m4x4 a, vec4 b, vec4 r);
void m4x4_rotate(m4x4 a, float angle, vec3 axis);

double deg_to_rad(double deg);
double rad_to_deg(double rad);

// helpers
void print_vec2(vec2 v);
void print_vec3(vec3 v);
void print_vec4(vec4 v);
void print_m2x2(m2x2 m);
void print_m2x3(m2x3 m);
void print_m2x4(m2x4 m);
void print_m3x2(m3x2 m);
void print_m3x3(m3x3 m);
void print_m3x4(m3x4 m);
void print_m4x2(m4x2 m);
void print_m4x3(m4x3 m);
void print_m4x4(m4x4 m);
void print_ymm(__m256 v);
void print_xmm(__m128 v);

// tranpose
void m2x2_transpose(m2x2 a, m2x2 r);
void m3x3_transpose(m3x3 a, m3x3 b);
void m4x4_transpose(m4x4 a, m4x4 b);
