#include "libmave.h"
#include <xmmintrin.h>

// vec2 operations.
void vec2_add(vec2 a, vec2 b, vec2 result) {
  _mm_storel_pi((__m64 *)result,
                _mm_add_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64 *)a),
                           _mm_loadl_pi(_mm_setzero_ps(), (__m64 *)b)));
}
void vec2_sub(vec2 a, vec2 b, vec2 result) {
  _mm_storel_pi((__m64 *)result,
                _mm_sub_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64 *)a),
                           _mm_loadl_pi(_mm_setzero_ps(), (__m64 *)b)));
}
void vec2_mul(vec2 a, vec2 b, vec2 result) {
  _mm_storel_pi((__m64 *)result,
                _mm_mul_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64 *)a),
                           _mm_loadl_pi(_mm_setzero_ps(), (__m64 *)b)));
}
void vec2_div(vec2 a, vec2 b, vec2 result) {
  _mm_storel_pi((__m64 *)result,
                _mm_div_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64 *)a),
                           _mm_loadl_pi(_mm_setzero_ps(), (__m64 *)b)));
}

// vec3 operations.
void vec3_add(vec3 a, vec3 b, vec3 result) {
  __m128 _result = _mm_add_ps(_mm_set_ps(0.0f, a[2], a[1], a[0]),
                              _mm_set_ps(0.0f, b[2], b[1], b[0]));
  _mm_storel_pi((__m64 *)result, _result);
  _mm_store_ss(result + 2,
               _mm_shuffle_ps(_result, _result, _MM_SHUFFLE(0, 0, 0, 2)));
}
void vec3_sub(vec3 a, vec3 b, vec3 result) {
  __m128 _result = _mm_sub_ps(_mm_set_ps(0.0f, a[2], a[1], a[0]),
                              _mm_set_ps(0.0f, b[2], b[1], b[0]));
  _mm_storel_pi((__m64 *)result, _result);
  _mm_store_ss(result + 2,
               _mm_shuffle_ps(_result, _result, _MM_SHUFFLE(0, 0, 0, 2)));
}
void vec3_mul(vec3 a, vec3 b, vec3 result) {
  __m128 _result = _mm_mul_ps(_mm_set_ps(0.0f, a[2], a[1], a[0]),
                              _mm_set_ps(0.0f, b[2], b[1], b[0]));
  _mm_storel_pi((__m64 *)result, _result);
  _mm_store_ss(result + 2,
               _mm_shuffle_ps(_result, _result, _MM_SHUFFLE(0, 0, 0, 2)));
}
void vec3_div(vec3 a, vec3 b, vec3 result) {
  __m128 _result = _mm_div_ps(_mm_set_ps(0.0f, a[2], a[1], a[0]),
                              _mm_set_ps(0.0f, b[2], b[1], b[0]));
  _mm_storel_pi((__m64 *)result, _result);
  _mm_store_ss(result + 2,
               _mm_shuffle_ps(_result, _result, _MM_SHUFFLE(0, 0, 0, 2)));
}

// vec4 operations.
void vec4_add(vec4 a, vec4 b, vec4 result) {
  _mm_store_ps(result, _mm_add_ps(_mm_load_ps(a), _mm_load_ps(b)));
}
void vec4_sub(vec4 a, vec4 b, vec4 result) {
  _mm_store_ps(result, _mm_sub_ps(_mm_load_ps(a), _mm_load_ps(b)));
}
void vec4_mul(vec4 a, vec4 b, vec4 result) {
  _mm_store_ps(result, _mm_mul_ps(_mm_load_ps(a), _mm_load_ps(b)));
}
void vec4_div(vec4 a, vec4 b, vec4 result) {
  _mm_store_ps(result, _mm_div_ps(_mm_load_ps(a), _mm_load_ps(b)));
}
