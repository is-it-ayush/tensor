#include "libmave.h"
#include <pmmintrin.h>
#include <stdio.h>
#include <immintrin.h>
#include <xmmintrin.h>

// helpers
void print_vec2(vec2 v) { printf("[%f, %f]\n", v[0], v[1]); }
void print_vec3(vec3 v) { printf("[%f, %f, %f]\n", v[0], v[1], v[2]); }
void print_vec4(vec4 v) {
  printf("[%f, %f, %f, %f]\n", v[0], v[1], v[2], v[3]);
}
void print_m2x2(m2x2 m) {
  print_vec2(m[0]);
  print_vec2(m[1]);
}
void print_m2x3(m2x3 m) {
  print_vec3(m[0]);
  print_vec3(m[1]);
}
void print_m2x4(m2x4 m) {
  print_vec4(m[0]);
  print_vec4(m[1]);
}
void print_ymm(__m256 v) {
  float *f = (float *)&v;
  printf("[%f, %f, %f, %f, %f, %f, %f, %f]\n", f[0], f[1], f[2], f[3], f[4],
         f[5], f[6], f[7]);
}
void print_xmm(__m128 v) {
  float *f = (float *)&v;
  printf("[%f, %f, %f, %f]\n", f[0], f[1], f[2], f[3]);
}

// transposes
void m2x2_transpose(m2x2 a, m2x2 r) {
  __m128 d = _mm_load_ps(a[0]);
  _mm_store_ps(r[0], _mm_shuffle_ps(d, d, _MM_SHUFFLE(3, 1, 2, 0)));
}
void m3x2_transpose(m3x2 a, m2x3 r) {
  __m256 x0 = _mm256_setzero_ps();
  x0 = _mm256_maskload_ps(a[0], _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1)); // 0 0 f e d c b a
  _mm256_maskstore_ps(r[0], _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1), _mm256_permutevar8x32_ps(x0, _mm256_set_epi32(7, 7, 5, 3, 1, 4, 2, 0)));
}
void m2x3_transpose(m2x3 a, m3x2 r) {
  __m256 x0 = _mm256_setzero_ps();
  x0 = _mm256_maskload_ps(a[0], _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1));
  _mm256_maskstore_ps(r[0], _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1), _mm256_permutevar8x32_ps(x0, _mm256_set_epi32(7, 7, 5, 2, 4, 1, 3, 0)));
}
void m2x4_transpose(m2x4 a, m4x2 b) {
  __m256 x0 = _mm256_load_ps(a[0]);
  _mm256_store_ps(b[0], _mm256_permutevar8x32_ps(x0, _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0)));
}

/**
 * General Operations: Add, Subtract, Multiply, Divide
 * Three kinds operations might exist,
 * - scalar-vector
 * - scalar-matrix
 * - vector-matrix
 * - matrix-vector
 * - matrix-matrix
 * - vector-vector (done)
 */

// operations
void vec2_add(vec2 a, vec2 b, vec2 r) {
  _mm_storel_pi((__m64 *)r,
                _mm_add_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64 *)a),
                           _mm_loadl_pi(_mm_setzero_ps(), (__m64 *)b)));
}
void vec2_sub(vec2 a, vec2 b, vec2 r) {
  _mm_storel_pi((__m64 *)r,
                _mm_sub_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64 *)a),
                           _mm_loadl_pi(_mm_setzero_ps(), (__m64 *)b)));
}
void vec2_mul(vec2 a, vec2 b, vec2 r) {
  _mm_storel_pi((__m64 *)r,
                _mm_mul_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64 *)a),
                           _mm_loadl_pi(_mm_setzero_ps(), (__m64 *)b)));
}
void vec2_div(vec2 a, vec2 b, vec2 r) {
  _mm_storel_pi((__m64 *)r,
                _mm_div_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64 *)a),
                           _mm_loadl_pi(_mm_setzero_ps(), (__m64 *)b)));
}

void vec3_add(vec3 a, vec3 b, vec3 r) {
  __m128 _result = _mm_add_ps(_mm_set_ps(0.0f, a[2], a[1], a[0]),
                              _mm_set_ps(0.0f, b[2], b[1], b[0]));
  _mm_storel_pi((__m64 *)r, _result);
  _mm_store_ss(r + 2,
               _mm_shuffle_ps(_result, _result, _MM_SHUFFLE(0, 0, 0, 2)));
}
void vec3_sub(vec3 a, vec3 b, vec3 r) {
  __m128 _result = _mm_sub_ps(_mm_set_ps(0.0f, a[2], a[1], a[0]),
                              _mm_set_ps(0.0f, b[2], b[1], b[0]));
  _mm_storel_pi((__m64 *)r, _result);
  _mm_store_ss(r + 2,
               _mm_shuffle_ps(_result, _result, _MM_SHUFFLE(0, 0, 0, 2)));
}
void vec3_mul(vec3 a, vec3 b, vec3 r) {
  __m128 _result = _mm_mul_ps(_mm_set_ps(0.0f, a[2], a[1], a[0]),
                              _mm_set_ps(0.0f, b[2], b[1], b[0]));
  _mm_storel_pi((__m64 *)r, _result);
  _mm_store_ss(r + 2,
               _mm_shuffle_ps(_result, _result, _MM_SHUFFLE(0, 0, 0, 2)));
}
void vec3_div(vec3 a, vec3 b, vec3 r) {
  __m128 _result = _mm_div_ps(_mm_set_ps(0.0f, a[2], a[1], a[0]),
                              _mm_set_ps(0.0f, b[2], b[1], b[0]));
  _mm_storel_pi((__m64 *)r, _result);
  _mm_store_ss(r + 2,
               _mm_shuffle_ps(_result, _result, _MM_SHUFFLE(0, 0, 0, 2)));
}

void vec4_add(vec4 a, vec4 b, vec4 r) {
  _mm_store_ps(r, _mm_add_ps(_mm_load_ps(a), _mm_load_ps(b)));
}
void vec4_sub(vec4 a, vec4 b, vec4 r) {
  _mm_store_ps(r, _mm_sub_ps(_mm_load_ps(a), _mm_load_ps(b)));
}
void vec4_mul(vec4 a, vec4 b, vec4 r) {
  _mm_store_ps(r, _mm_mul_ps(_mm_load_ps(a), _mm_load_ps(b)));
}
void vec4_div(vec4 a, vec4 b, vec4 r) {
  _mm_store_ps(r, _mm_div_ps(_mm_load_ps(a), _mm_load_ps(b)));
}

void m2x2_add(m2x2 a, m2x2 b, m2x2 r) {
  vec2_add(a[0], b[0], r[0]);
  vec2_add(a[1], b[1], r[1]);
}
void m2x2_sub(m2x2 a, m2x2 b, m2x2 r) {
  vec2_sub(a[0], b[0], r[0]);
  vec2_sub(a[1], b[1], r[1]);
}
void m2x2_mul(m2x2 a, m2x2 b, m2x2 r) {
  __m128 x0, x1, x2, x3, x4;

  // a b  x  e f  =  ae+bg af+bh
  // c d     g h     ce+dg cf+dh

  x1 = _mm_load_ps(a[0]); // d c b a | 3 2 1 0 | 128 64 32 16
  x2 = _mm_load_ps(b[0]); // h g f e | 3 2 1 0 | 128 64 32 16

  x3 = _mm_shuffle_ps(x2, x2, _MM_SHUFFLE(1, 0, 1, 0)); // f e f e
  x4 = _mm_shuffle_ps(x2, x2, _MM_SHUFFLE(3, 2, 3, 2)); // h g h g

  x0 = _mm_shuffle_ps(x1, x1, _MM_SHUFFLE(2, 2, 0, 0)); // c c a a
  x2 = _mm_shuffle_ps(x1, x1, _MM_SHUFFLE(3, 3, 1, 1)); // d d b b

  x0 = _mm_add_ps(_mm_mul_ps(x0, x3),
                  _mm_mul_ps(x2, x4)); // ae+bg af+bh ce+dg cf+dh

  _mm_store_ps(r[0], x0);
}
void m2x2_vec2_mul(m2x2 a, vec2 b, vec2 r) {
  __m128 x0, x1, x2, x3;

  // a b  x  e  =  ae+bf
  // c d     f     ce+df

  x1 = _mm_load_ps(a[0]); // d c b a | 3 2 1 0 | 128 64 32 16
  x2 = _mm_loadl_pi(_mm_setzero_ps(),
                    (__m64 *)b); // 0 0 f e | 3 2 1 0 | 128 64 32 16

  x3 = _mm_shuffle_ps(x2, x2, _MM_SHUFFLE(1, 0, 1, 0)); // f e f e
  x3 = _mm_mul_ps(x1, x3); // df ce bf ae  (fefe * dcba)
  x0 = _mm_shuffle_ps(
      x3, x3,
      _MM_SHUFFLE(0, 3, 0, 1)); // ae df ae bf (rearrange from df ce bf ae)
  x0 = _mm_add_ps(x3,
                  x0); // (df ce bf ae) * (ae df ae bf) = [-, ce+df, -, ae+bf]
  x0 = _mm_shuffle_ps(x0, x0, _MM_SHUFFLE(0, 0, 2, 0)); // [-,  -, ce+df, ae+bf]

  _mm_storel_pi((__m64 *)r, x0);
}
void m2x2_m2x3_mul(m2x2 a, m2x3 b, m2x3 r) {
  // a b x e f g = ae+bh af+bi ag+bj
  // c d   h i j   ce+dh cf+di cg+dj

  // transposed b:
  // e h
  // f i
  // g j

  __m256 y0, y1 = _mm256_setzero_ps(), y2, y3;
  __m128 lo, hi, sum;
  m3x2 b_t;

  m2x3_transpose(b, b_t);

  y0 = _mm256_broadcast_ps((__m128 *)a);                                           // d c b a d c b a
  y1 = _mm256_maskload_ps(b_t[0], _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1)); // 0 0 j g i f h e
  y1 = _mm256_permutevar8x32_ps(y1, _mm256_set_epi32(1, 0, 5, 4, 3, 2, 1, 0));     // h e j g i f h e

  y2 = _mm256_mul_ps(y0, y1); // dh ce bj ag di cf bh ae
  lo = _mm256_extractf128_ps(y2, 0); // di cf bh ae
  hi = _mm256_extractf128_ps(y2, 1); // dh ce bj ag

  sum = _mm_hadd_ps(lo, hi); // ce+dh ag+bj cf+di bh+ae

  y3 = _mm256_insertf128_ps(_mm256_setzero_ps(), sum, 0); // 0 0 0 0 ce+dh ag+bj cf+di bh+ae

  y1 = _mm256_permutevar8x32_ps(y1, _mm256_set_epi32(5, 4, 3, 2, 5, 4, 3, 2)); // j g i f j g i f
  y2 = _mm256_mul_ps(y0, y1); // dj cg bi af dj cg bi af
  lo = _mm256_extractf128_ps(y2, 0); // dj cg bi af
  sum = _mm_hadd_ps(lo, lo); // cg+dj af+bi cg+dj af+bi

  y3 = _mm256_insertf128_ps(y3, sum, 1); // cg+dj af+bi cg+dj af+bi ce+dh ag+bj cf+di bh+ae
  y3 = _mm256_permutevar8x32_ps(y3, _mm256_set_epi32(-1, -1, 5, 1, 3, 2, 4, 0)); // ae+bf af+bi ag+bj ce+dh cf+di cg+dj

  _mm256_maskstore_ps(r[0], _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1), y3);
}
void m2x2_m2x4_mul(m2x2 a, m2x4 b, m2x4 r) {
  // a b x e f g h = ae+bi af+bj ag+bk ah+bl
  // c d   i j k l   ce+di cf+dj cg+dk ch+dl

  // transposed b:
  // e i
  // f j
  // g k
  // h l

  __m256 y0 = _mm256_setzero_ps(), y1, y2, y3;
  __m128 lo, hi, sum;
  m4x2 b_t;
  m2x4_transpose(b, b_t);

  y0 = _mm256_broadcast_ps((__m128 *)a); // d c b a d c b a
  y1 = _mm256_load_ps(b_t[0]);           // l h k g j f i e

  y2 = _mm256_mul_ps(y0, y1); // dl ch bk ag dj cf bi ae
  lo = _mm256_extractf128_ps(y2, 0); // dj cf bi ae
  hi = _mm256_extractf128_ps(y2, 1); // dl ch bk ag
  sum = _mm_hadd_ps(lo, hi); // ch+dl ag+bk cf+dj bi+ae

  y3 = _mm256_insertf128_ps(_mm256_setzero_ps(), sum, 0); // 0 0 0 0 ch+dl ag+bk cf+dj bi+ae

  y0 = _mm256_permutevar8x32_ps(y0, _mm256_set_epi32(1, 0, 3, 2, 1, 0, 3, 2)); // b a d c b a d c
  y2 = _mm256_mul_ps(y0, y1); // bl ah dk cg bj af di ce
  lo = _mm256_extractf128_ps(y2, 0); // bj af di ce
  hi = _mm256_extractf128_ps(y2, 1); // bl ah dk cg
  sum = _mm_hadd_ps(lo, hi); // ah+bl cg+dk af+bj ce+di

  y3 = _mm256_insertf128_ps(y3, sum, 1); // ah+bl cg+dk af+bj ce+di ch+dl ag+bk cf+dj bi+ae

  y3 = _mm256_permutevar8x32_ps(y3, _mm256_set_epi32(3, 6, 1, 4, 7, 2, 5, 0)); // ae+bi af+bj ag+bk ah+bl ce+di cf+dj cg+dk ch+dl

  _mm256_store_ps(r[0], y3);
}

void m2x3_add(m2x3 a, m2x3 b, m2x3 r) {
  vec3_add(a[0], b[0], r[0]);
  vec3_add(a[1], b[1], r[1]);
}
void m2x3_sub(m2x3 a, m2x3 b, m2x3 r) {
  vec3_sub(a[0], b[0], r[0]);
  vec3_sub(a[1], b[1], r[1]);
}
void m2x3_vec3_mul(m2x3 a, vec3 b, vec2 r) {
  // a b c  x  g  = ag+bh+ci
  // d e f     h    dg+eh+fi
  //           i

  __m256 x0 = _mm256_setzero_ps(), x1 = _mm256_setzero_ps(), x2, x3;
  __m128 lo, hi, sum;
  x0 = _mm256_maskload_ps(a[0], _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1)); // _ _ f e d c b a
  x1 = _mm256_maskload_ps(b, _mm256_set_epi32(0, 0,  0,  0,  0, -1, -1, -1));    // _ _ _ _ _ i h g

  x2 = _mm256_mul_ps(x0, x1); // _ _ _ _ _ ci bh ag
  lo = _mm256_extractf128_ps(x2, 0); // _ ci bh ag
  hi = _mm256_extractf128_ps(x2, 1); // _  _  _  _
  sum = _mm_hadd_ps(lo, hi); //
  sum = _mm_hadd_ps(sum, sum);
  _mm_store_ss(r, sum);

  x0 = _mm256_permutevar8x32_ps(x0, _mm256_set_epi32(0, 0, 0, 0, 0, 5, 4, 3)); // _ _ _ _ _ f e d
  x3 = _mm256_mul_ps(x0, x1); // _ _ _ _ _ fi eh dg : we'll unload this to r[1]
  lo = _mm256_extractf128_ps(x3, 0); // _ fi eh dg
  sum = _mm_hadd_ps(lo, hi); //
  sum = _mm_hadd_ps(sum, sum);
  _mm_store_ss(r+1, sum);
}
// todo: optimize...
void m2x3_m3x2_mul(m2x3 a, m3x2 b, m2x2 r) {
  // a b c     g h   ag+bi+ck ah+bj+cl
  // d e f  x  i j = dg+ei+fk dh+ej+fl
  //           k l

  __m256 y0 = _mm256_setzero_ps(), y1 = _mm256_setzero_ps(), y2;
  __m128 lo, hi, sum, zero = _mm_setzero_ps();
  m2x3 b_t;

  // Transpose b
  m3x2_transpose(b, b_t); // 0 0 l j h k i g

  y0 = _mm256_maskload_ps(a[0], _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1));   // 0 0 f e d c b a
  y1 = _mm256_maskload_ps(b_t[0], _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1)); // 0 0 l j h k i g

  y2 = _mm256_mul_ps(y0, y1); // 0 0 fl ej dh ck bi ag
  y2 = _mm256_permutevar8x32_ps(y2, _mm256_set_epi32(7, 5, 4, 3, 7, 2, 1, 0)); // 0 fl ej dh 0 ck bi ag

  // r[0][0]
  lo = _mm256_extractf128_ps(y2, 0); // 0 ck bi ag
  sum = _mm_hadd_ps(lo, zero); // 0 0 ck ag+bi
  sum = _mm_hadd_ps(sum, sum); // 0 0 0 ag+bi+ck
  _mm_store_ss(&r[0][0], sum);

  // r[1][1]
  hi = _mm256_extractf128_ps(y2, 1); // 0 fl ej dh
  sum = _mm_hadd_ps(hi, zero); // 0 0 fl dh+ej
  sum = _mm_hadd_ps(sum, sum); // 0 0 0 dh+ej+fl
  _mm_store_ss(&r[1][1], sum);

  y0 = _mm256_permutevar8x32_ps(y0, _mm256_set_epi32(7, 7, 2, 1, 0, 5, 4, 3)); // 0 0 c b a f e d
  y2 = _mm256_mul_ps(y0, y1); // 0 0 cl bj ah fk ei dg
  y2 = _mm256_permutevar8x32_ps(y2, _mm256_set_epi32(7, 5, 4, 3, 7, 2, 1, 0)); // 0 cl bj ah 0 fk ei dg

  // r[1][0]
  lo = _mm256_extractf128_ps(y2, 0); // 0 fk ei dg
  sum = _mm_hadd_ps(lo, zero); // 0 0 fk dg+ei
  sum = _mm_hadd_ps(sum, sum); // 0 0 0 dg+ei+fk
  _mm_store_ss(&r[1][0], sum);

  // r[0][1]
  hi = _mm256_extractf128_ps(y2, 1); // 0 cl bj ah
  sum = _mm_hadd_ps(hi, zero); // 0 0 cl ah+bj
  sum = _mm_hadd_ps(sum, sum); // 0 0 0 ah+bj+cl
  _mm_store_ss(&r[0][1], sum);
}
void m2x3_m3x3_mul(m2x3 a, m3x3 b, m2x3 r) {
  // a b c     g h i   ag+bj+cm ah+bk+cn ai+bl+co
  // d e f  x  j k l = dg+ej+fm dh+ek+fn di+el+fo
  //           m n o

  __m256 y0 = _mm256_setzero_ps(), y1 = _mm256_setzero_ps(), y2;
  __m128 lo, hi, sum;

  y0 = _mm256_maskload_ps(a[0], _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1)); // 0 0 f e d c b a

  // o l i n k h m j g
  // 0 0   f  e  d  c  b  a -- maskload, ymm0
  // 0 0   n  k  h  m  j  g -- maskload, ymm1
  // 0 0  fn ek dh cm bj ag -- multiply, ymm2
  //
  // 0 fn ek dh 0 cm bj ag -- permute, ymm2
  // 0 cm bj ag -- extract low, xmm0
  // cm bj+ag cm bj+ag - hadd, xmm0
  // bj+ag+cm bj+ag+cm bj+ag+cm bj+ag+cm - hadd, xmm0
  //
  // 0 0 0 0 0 0 0 0 -- new zero inited, ymm3
  // 0 0 0 0 bj+ag+cm bj+ag+cm bj+ag+cm bj+ag+cm -- insertf128, ymm3 (low)
  // 0 0 0 0 0 0 0 bj+ag+cm -- permute, ymm3
  //
  // 0 fn ek dh -- extract high, xmm0
  // fn ek+dh fn ek+dh - hadd, xmm0
  // ek+dh+fn ek+dh+fn ek+dh+fn ek+dh+fn - hadd, xmm0
  //
  // ek+dh+fn ek+dh+fn ek+dh+fn ek+dh+fn 0 0 0 bg+ag+cm -- insertf128, ymm3 (high)
  // 0 0 0 0 0 0 ek+dh+fn bg+ag+cm -- permute, ymm3
  //
  //
  // 0  0   f  e   d   c  b  a -- NOP (no instruction since we already have ymm0)
  // 0  0
  // 0  0   m  j   g   o  l  i -- permute, ymm1
  // 0  0  fm ej  dg  co bl ai -- multiply, ymm2
  // 0 fm  ej dg   0  co bl ai -- permute, ymm2
  // 0 co bl ai -- extract low, xmm0
  // co bl+ai co bl+ai -- hadd, xmm0
  // bl+ai+co bl+ai+co bl+ai+co bl+ai+co -- hadd, xmm0
  //
  // bl+ai+co bl+ai+co bl+ai+co bl+ai+co 0 0 ek+dh+fn bg+ag+cm -- insertf128, ymm3 (low)
  // 0 0 0 0 0 bl+ai+co ek+dh+fn bg+ag+cm -- permute, ymm3
  //
  // 0 fm ej dg -- extract high, xmm0
  // fm ej+dg fm ej+dg -- hadd, xmm0
  // ej+dg+fm ej+dg+fm ej+dg+fm ej+dg+fm -- hadd, xmm0
  //
  // ej+dg+fm ej+dg+fm ej+dg+fm ej+dg+fm 0 bl+ai+co ek+dh+fn bg+ag+cm -- insertf128, ymm3 (high)
  // 0 0 0 0 0 0 ej+dg+fm bl+ai+co ek+dh+fn bg+ag+cm -- permute, ymm3
  //
  // 0 0  f  e   d   c  b  a -- NOP (no instruction since we already have ymm0)
  //
  // 0 0  o  l   i   n  k  h --
  // 0 0 fo el  di  cn bk ah

  // 0 fo el di 0 cn bk ah

}


// 2x2 * 2x1 (m2x2_vec2_mul) | 2x2 * 2x2 (m2x2_mul) | 2x2 * 2x3 (m2x2_m2x3_mul) | 2x2 * 2x4 (m2x2_m2x4_mul)
// 2x3 * 3x1 (m2x3_vec3_mul) | 2x3 * 3x2 (m2x3_m3x2_mul) | 2x3 * 3x3 | 2x3 * 3x4
// 2x4 * 4x1 | 2x4 * 4x2 | 2x4 * 4x3 | 2x4 * 4x4
// 3x2 * 2x1
// 3x3 * 3x1 |   -       | 3x3 * 3x3 |  -
// 3x4 * 4x1
// 4x2 * 2x1
// 4x3 * 3x1
// 4x4 * 4x1 |   -       |    -      | 4x4 * 4x4

// mat * vec | mat * mat | mat * mat | mat * mat
