#include <assert.h>
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <xmmintrin.h>

#ifndef T_ALIGNMENT
#define T_ALIGNMENT(X) __attribute((aligned(X)))
#ifdef __AVX__
#define T_MEMALIGN T_ALIGNMENT(32)
#elif __SSE__
#define T_MEMALIGN T_ALIGNMENT(16)
#endif
#endif

#ifndef T_SCOPE
#define T_SCOPE extern
#endif

#ifndef T_CONSTANTS
#define MATH_PI 3.1415926535897932384626433
#define EPSILON 1e-6
#define T_CONSTANTS
#endif

/***
 * ----
 * Definitions
 * ----
 */

/**
 * A two-dimensional vector represented by [[a, b]].
 */
typedef float vec2[2];
/**
 * A three-dimensional vector represented by [[a, b, c]].
 */
typedef float vec3[3];
/**
 * A four-dimensional vector represented by [[a, b, c, d]].
 */
typedef float vec4[4];

/**
 * A 2x2 matrix represented by [[a, b], [c, d]].
 */
typedef T_MEMALIGN vec2 m2x2[2];
/**
 * A 2x3 matrix represented by [[a, b, c], [d, e, f]].
 */
typedef T_MEMALIGN vec3 m2x3[2];
/**
 * A 2x4 matrix represented by [[a, b, c, d], [e, f, g, h]].
 */
typedef T_MEMALIGN vec4 m2x4[2];

/**
 * A 3x2 matrix represented by [[a, b], [c, d], [e, f]].
 */
typedef T_MEMALIGN vec2 m3x2[3];
/**
 * A 3x3 matrix represented by [[a, b, c], [d, e, f], [g, h, i]].
 */
typedef T_MEMALIGN vec3 m3x3[3];
/**
 * A 3x4 matrix represented by [[a, b, c, d], [e, f, g, h], [i, j, k, l]].
 */
typedef T_MEMALIGN vec4 m3x4[3];

/**
 * A 4x2 matrix represented by [[a, b], [c, d], [e, f], [g, h]].
 */
typedef T_MEMALIGN vec2 m4x2[4];
/**
 * A 4x3 matrix represented by [[a, b, c], [d, e, f], [g, h, i], [j, k, l]].
 */
typedef T_MEMALIGN vec3 m4x3[4];
/**
 * A 4x4 matrix represented by [[a, b, c, d], [e, f, g, h], [i, j, k, l], [m, n,
 * o, p]].
 */
typedef T_MEMALIGN vec4 m4x4[4];

/***
 * ----
 * Functions
 * ----
 */

/**
 * Adds two vectors `a` and `b` and stores the result in `r`.
 */
T_SCOPE void vec2_add(vec2 a, vec2 b, vec2 r);
/**
 * Subtracts two vectors `a` and `b` and stores the result in `r`.
 */
T_SCOPE void vec2_sub(vec2 a, vec2 b, vec2 r);
/**
 * Scales two vectors `a` and `b` and stores the result in `r`.
 */
T_SCOPE void vec2_scale(vec2 a, vec2 b, vec2 r);
/**
 * Calculates the dot product of two vectors `a` and `b`.
 */
T_SCOPE float vec2_dot_mul(vec2 a, vec2 b);
/**
 * Calculates the determinent of two vectors `a` and `b`.
 */
T_SCOPE float vec2_determinent(vec2 a, vec2 b);
/**
 * Divides two vectors `a` and `b` and stores the result in `r`.
 */
T_SCOPE void vec2_div(vec2 a, vec2 b, vec2 r);
/**
 * Calculates the magnitude of a vector `a`.
 */
T_SCOPE float vec2_mag(vec2 a);

/**
 * Adds two vectors `a` and `b` and stores the result in `r`.
 */
T_SCOPE void vec3_add(vec3 a, vec3 b, vec3 r);
/**
 * Subtracts two vectors `a` and `b` and stores the result in `r`.
 */
T_SCOPE void vec3_sub(vec3 a, vec3 b, vec3 r);
/**
 * Scales a vector `a` by a scalar `b` and stores the result in `r`.
 */
T_SCOPE void vec3_scale(vec3 a, float b, vec3 r);
/**
 * Calculates the dot product of two vectors `a` and `b`.
 */
T_SCOPE float vec3_dot_mul(vec3 a, vec3 b);
/**
 * Calculates the cross product of two vectors `a` and `b`.
 */
T_SCOPE void vec3_cross_mul(vec3 a, vec3 b, vec3 r);
/**
 * Divides two vectors `a` and `b` and stores the result in `r`.
 */
T_SCOPE void vec3_div(vec3 a, vec3 b, vec3 r);
/**
 * Calculates the magnitude of a vector `a`.
 */
T_SCOPE float vec3_mag(vec3 a);
/*
 * Normalizes a vector `a` and stores the result in `r`.
 */
T_SCOPE void vec3_normalize(vec3 a, vec3 r);

/**
 * Adds two vectors `a` and `b` and stores the result in `r`.
 */
T_SCOPE void vec4_add(vec4 a, vec4 b, vec4 r);
/**
 * Subtracts two vectors `a` and `b` and stores the result in `r`.
 */
T_SCOPE void vec4_sub(vec4 a, vec4 b, vec4 r);
/**
 * Scales two vectors `a` and `b` and stores the result in `r`.
 */
T_SCOPE void vec4_scale(vec4 a, vec3 by, vec4 r);
/**
 * Calculates the dot product of two vectors `a` and `b`.
 */
T_SCOPE float vec4_dot_mul(vec4 a, vec4 b);
/**
 * Divides two vectors `a` and `b` and stores the result in `r`.
 */
T_SCOPE void vec4_div(vec4 a, vec4 b, vec4 r);
/**
 * Calculates the magnitude of a vector `a`.
 */
T_SCOPE float vec4_mag(vec4 a);
/**
 * Rotates a vector `a` by an angle `angle` around an axis `axis` and stores the
 * result in `r`.
 */
T_SCOPE void vec4_rotate(vec4 a, float angle, vec3 axis, vec4 r);
/*
 * Translates a vector `a` by a vector `by` and stores the result in `r`.
 * The vector `by` is a 3D vector that represents the translation
 * on the x, y, and z axes.
 */
T_SCOPE void vec4_translate(vec4 a, vec3 by, vec4 r);
/**
 * Scales a vector `a` by a vector `by` and stores the result in `r`.
 * The vector `by` is a 3D vector that represents the scaling on the x, y, and z
 * axes.
 */
T_SCOPE void vec4_scale(vec4 a, vec3 by, vec4 r);

/**
 * Adds two 2x2 matrices `a` and `b` and stores the resulting 2x2 matrix in `r`.
 */
T_SCOPE void m2x2_add(m2x2 a, m2x2 b, m2x2 r);
/**
 * Subtracts two 2x2 matrices `a` and `b` and stores the resulting 2x2 matrix in
 * `r`.
 */
T_SCOPE void m2x2_sub(m2x2 a, m2x2 b, m2x2 r);
/**
 * Multiplies two 2x2 matrices `a` and `b` and stores the resulting 2x2 matrix
 * in `r`.
 */
T_SCOPE void m2x2_mul(m2x2 a, m2x2 b, m2x2 r);
/**
 * Multiplies a 2x2 matrix `a` and a 2D vector `b` and stores the resulting 2D
 * vector in `r`.
 */
T_SCOPE void m2x2_vec2_mul(m2x2 a, vec2 b, vec2 r);

/**
 * Adds two 2x3 matrices `a` and `b` and stores the resulting 2x3 matrix in `r`.
 */
T_SCOPE void m2x3_add(m2x3 a, m2x3 b, m2x3 r);
/**
 * Subtracts two 2x3 matrices `a` and `b` and stores the resulting 2x3 matrix in
 * `r`.
 */
T_SCOPE void m2x3_sub(m2x3 a, m2x3 b, m2x3 r);
/**
 * Multiplies a 2x3 matrix `a` and a 3D vector `b` and stores the resulting 2D
 * vector in `r`.
 */
T_SCOPE void m2x3_vec3_mul(m2x3 a, vec3 b, vec2 r);

/**
 * Adds two 2x4 matrices `a` and `b` and stores the resulting 2x4 matrix in `r`.
 */
T_SCOPE void m2x4_add(m2x4 a, m2x4 b, m2x4 r);
/**
 * Subtracts two 2x4 matrices `a` and `b` and stores the resulting 2x4 matrix in
 * `r`.
 */
T_SCOPE void m2x4_sub(m2x4 a, m2x4 b, m2x4 r);
/**
 * Multiplies a 2x4 matrix `a` and a 4D vector `b` and stores the resulting 2D
 * vector in `r`.
 */
T_SCOPE void m2x4_vec4_mul(m2x4 a, vec4 b, vec2 r);

/**
 * Adds two 3x2 matrices `a` and `b` and stores the resulting 3x2 matrix in `r`.
 */
T_SCOPE void m3x2_add(m3x2 a, m3x2 b, m3x2 r);
/**
 * Subtracts two 3x2 matrices `a` and `b` and stores the resulting 3x2 matrix in
 * `r`.
 */
T_SCOPE void m3x2_sub(m3x2 a, m3x2 b, m3x2 r);
/**
 * Multiplies a 3x2 matrix `a` and a 2D vector `b` and stores the resulting 3D
 * vector in `r`.
 */
T_SCOPE void m3x2_vec2_mul(m3x2 a, vec2 b, vec3 r);

/**
 * Adds two 3x3 matrices `a` and `b` and stores the resulting 3x3 matrix in `r`.
 */
T_SCOPE void m3x3_add(m3x3 a, m3x3 b, m3x3 r);
/**
 * Subtracts two 3x3 matrices `a` and `b` and stores the resulting 3x3 matrix in
 * `r`.
 */
T_SCOPE void m3x3_sub(m3x3 a, m3x3 b, m3x3 r);
/**
 * Multiplies two 3x3 matrices `a` and `b` and stores the resulting 3x3 matrix
 * in `r`.
 */
T_SCOPE void m3x3_mul(m3x3 a, m3x3 b, m3x3 r);
/**
 * Multiplies a 3x3 matrix `a` and a 3D vector `b` and stores the resulting 3D
 * vector in `r`.
 */
T_SCOPE void m3x3_vec3_mul(m3x3 a, vec3 b, vec3 r);

/**
 * Adds two 3x4 matrices `a` and `b` and stores the resulting 3x4 matrix in `r`.
 */
T_SCOPE void m3x4_add(m3x4 a, m3x4 b, m3x4 r);
/**
 * Subtracts two 3x4 matrices `a` and `b` and stores the resulting 3x4 matrix in
 * `r`.
 */
T_SCOPE void m3x4_sub(m3x4 a, m3x4 b, m3x4 r);
/**
 * Multiplies a 3x4 matrix `a` and a 4D vector `b` and stores the resulting 3D
 * vector in `r`.
 */
T_SCOPE void m3x4_vec4_mul(m3x4 a, vec4 b, vec3 r);

/**
 * Adds two 4x2 matrices `a` and `b` and stores the resulting 4x2 matrix in `r`.
 */
T_SCOPE void m4x2_add(m4x2 a, m4x2 b, m4x2 r);
/**
 * Subtracts two 4x2 matrices `a` and `b` and stores the resulting 4x2 matrix in
 * `r`.
 */
T_SCOPE void m4x2_sub(m4x2 a, m4x2 b, m4x2 r);
/**
 * Multiplies a 4x2 matrix `a` and a 2D vector `b` and stores the resulting 4D
 * vector in `r`.
 */
T_SCOPE void m4x2_vec2_mul(m4x2 a, vec2 b, vec4 r);

/**
 * Adds two 4x3 matrices `a` and `b` and stores the resulting 4x3 matrix in `r`.
 */
T_SCOPE void m4x3_add(m4x3 a, m4x3 b, m4x3 r);
/**
 * Subtracts two 4x3 matrices `a` and `b` and stores the resulting 4x3 matrix in
 * `r`.
 */
T_SCOPE void m4x3_sub(m4x3 a, m4x3 b, m4x3 r);
/**
 * Multiplies a 4x3 matrix `a` and a 3D vector `b` and stores the resulting 4D
 * vector in `r`.
 */
T_SCOPE void m4x3_vec3_mul(m4x3 a, vec3 b, vec4 r);

/**
 * Adds two 4x4 matrices `a` and `b` and stores the resulting 4x4 matrix in `r`.
 */
T_SCOPE void m4x4_add(m4x4 a, m4x4 b, m4x4 r);
/**
 * Subtracts two 4x4 matrices `a` and `b` and stores the resulting 4x4 matrix in
 * `r`.
 */
T_SCOPE void m4x4_sub(m4x4 a, m4x4 b, m4x4 r);
/**
 * Multiplies two 4x4 matrices `a` and `b` and stores the resulting 4x4 matrix
 * in `r`.
 */
T_SCOPE void m4x4_mul(m4x4 a, m4x4 b, m4x4 r);
/**
 * Multiplies a 4x4 matrix `a` and a 4D vector `b` and stores the resulting 4D
 * vector in `r`.
 */
T_SCOPE void m4x4_vec4_mul(m4x4 a, vec4 b, vec4 r);
/*
 * Generates a 4x4 identity matrix.
 */
T_SCOPE void m4x4_identity(m4x4 a);
/**
 * Generates a 4x4 rotation matrix around an axis `axis` by an angle `angle` &
 * multiplies it with the matrix `a`. Use `a` as the identity matrix if you want
 * to generate a rotation matrix only.
 */
T_SCOPE void m4x4_make_rotate(m4x4 a, vec3 axis, float angle);
/*
 * Translates a 4x4 matrix `a` by a vector `t`.
 */
T_SCOPE void m4x4_make_translate(m4x4 a, vec3 t);
/*
 * Scales a 4x4 matrix `a` by a vector `s`.
 */
T_SCOPE void m4x4_make_scale(m4x4 a, vec3 s);
/**
 * Generates a 4x4 perspective matrix using the field of view `fov_x`, aspect
 * ratio `aspect_ratio`, near plane `n`, and far plane `f` and stores it after
 * multiplying it with the matrix `a`. The field of view is in degrees.
 */
T_SCOPE void m4x4_make_perspective(
    m4x4 a, float fov_x, float aspect_ratio, float n, float f
);
/*
 * Generates a camera view matrix using the camera position, target
 * and the up vector.
 */
T_SCOPE void m4x4_make_look_at(m4x4 a, vec3 position, vec3 target, vec3 up);

/**
 * Converts degrees to radians.
 */
T_SCOPE double deg_to_rad(double deg);
/**
 * Converts radians to degrees.
 */
T_SCOPE double rad_to_deg(double rad);

// helpers

T_SCOPE void print_vec2(vec2 v);
T_SCOPE void print_vec3(vec3 v);
T_SCOPE void print_vec4(vec4 v);
T_SCOPE void print_m2x2(m2x2 m);
T_SCOPE void print_m2x3(m2x3 m);
T_SCOPE void print_m2x4(m2x4 m);
T_SCOPE void print_m3x2(m3x2 m);
T_SCOPE void print_m3x3(m3x3 m);
T_SCOPE void print_m3x4(m3x4 m);
T_SCOPE void print_m4x2(m4x2 m);
T_SCOPE void print_m4x3(m4x3 m);
T_SCOPE void print_m4x4(m4x4 m);
T_SCOPE void print_ymm(__m256 v);
T_SCOPE void print_xmm(__m128 v);

// tranpose
T_SCOPE void m2x2_transpose(m2x2 a, m2x2 r);
T_SCOPE void m3x3_transpose(m3x3 a, m3x3 b);
T_SCOPE void m4x4_transpose(m4x4 a, m4x4 b);

#define TENSOR_IMPLEMENTATION // comment this line to include the
// implementation
#ifdef TENSOR_IMPLEMENTATION

/***
 * ----
 * Implementation
 * ----
 */

T_SCOPE void print_vec2(vec2 v) {
  printf("[%f, %f]\n", v[0], v[1]);
}
T_SCOPE void print_vec3(vec3 v) {
  printf("[%f, %f, %f]\n", v[0], v[1], v[2]);
}
T_SCOPE void print_vec4(vec4 v) {
  printf("[%f, %f, %f, %f]\n", v[0], v[1], v[2], v[3]);
}
T_SCOPE void print_m2x2(m2x2 m) {
  print_vec2(m[0]);
  print_vec2(m[1]);
}
T_SCOPE void print_m2x3(m2x3 m) {
  print_vec3(m[0]);
  print_vec3(m[1]);
}
T_SCOPE void print_m2x4(m2x4 m) {
  print_vec4(m[0]);
  print_vec4(m[1]);
}
T_SCOPE void print_m3x2(m3x2 m) {
  print_vec2(m[0]);
  print_vec2(m[1]);
  print_vec2(m[2]);
}
T_SCOPE void print_m3x3(m3x3 m) {
  print_vec3(m[0]);
  print_vec3(m[1]);
  print_vec3(m[2]);
}
T_SCOPE void print_m3x4(m3x4 m) {
  print_vec4(m[0]);
  print_vec4(m[1]);
  print_vec4(m[2]);
}
T_SCOPE void print_m4x2(m4x2 m) {
  print_vec2(m[0]);
  print_vec2(m[1]);
  print_vec2(m[2]);
  print_vec2(m[3]);
}
T_SCOPE void print_m4x3(m4x3 m) {
  print_vec3(m[0]);
  print_vec3(m[1]);
  print_vec3(m[2]);
  print_vec3(m[3]);
}
T_SCOPE void print_m4x4(m4x4 m) {
  print_vec4(m[0]);
  print_vec4(m[1]);
  print_vec4(m[2]);
  print_vec4(m[3]);
}
T_SCOPE void print_ymm(__m256 v) {
  float* f = (float*)&v;
  printf(
      "[%f, %f, %f, %f, %f, %f, %f, %f]\n", f[0], f[1], f[2], f[3], f[4], f[5],
      f[6], f[7]
  );
}
T_SCOPE void print_xmm(__m128 v) {
  float* f = (float*)&v;
  printf("[%f, %f, %f, %f]\n", f[0], f[1], f[2], f[3]);
}

// transposes
T_SCOPE void m2x2_transpose(m2x2 a, m2x2 r) {
  __m128 d = _mm_load_ps(a[0]);
  _mm_store_ps(r[0], _mm_shuffle_ps(d, d, _MM_SHUFFLE(3, 1, 2, 0)));
}
T_SCOPE void m3x3_transpose(m3x3 a, m3x3 b) {
  _mm256_store_ps(
      b[0], _mm256_permutevar8x32_ps(
                _mm256_load_ps(a[0]), _mm256_set_epi32(5, 2, 7, 4, 1, 6, 3, 0)
            )
  );
  b[2][2] = a[2][2]; // :( don't have access to avx512
}
T_SCOPE void m4x4_transpose(m4x4 a, m4x4 b) {
  // a b c d     a e i m
  // e f g h  =  b f j n
  // i j k l     c g k o
  // m n o p     d h l p

  // i don't like this solution at all.
  // using special instructions doesn't make sense here...
  for(int i = 0; i < 4; i++) {
    for(int j = 0; j < 4; j++) {
      b[i][j] = a[j][i];
    }
  }
}

T_SCOPE void vec2_add(vec2 a, vec2 b, vec2 r) {
  _mm_storel_pi(
      (__m64*)r, _mm_add_ps(
                     _mm_loadl_pi(_mm_setzero_ps(), (__m64*)a),
                     _mm_loadl_pi(_mm_setzero_ps(), (__m64*)b)
                 )
  );
}
T_SCOPE void vec2_sub(vec2 a, vec2 b, vec2 r) {
  _mm_storel_pi(
      (__m64*)r, _mm_sub_ps(
                     _mm_loadl_pi(_mm_setzero_ps(), (__m64*)a),
                     _mm_loadl_pi(_mm_setzero_ps(), (__m64*)b)
                 )
  );
}
T_SCOPE void vec2_scale(vec2 a, vec2 b, vec2 r) {
  _mm_storel_pi(
      (__m64*)r, _mm_mul_ps(
                     _mm_loadl_pi(_mm_setzero_ps(), (__m64*)a),
                     _mm_loadl_pi(_mm_setzero_ps(), (__m64*)b)
                 )
  );
}
T_SCOPE float vec2_dot_mul(vec2 a, vec2 b) {
  return _mm_cvtss_f32(_mm_dp_ps(
      _mm_loadl_pi(_mm_setzero_ps(), (__m64*)a),
      _mm_loadl_pi(_mm_setzero_ps(), (__m64*)b), 0b00110001
  ));
}
T_SCOPE float vec2_determinent(vec2 a, vec2 b) {
  // a  x  p  = aq - bp
  // b     q
  __m128 x0;
  x0 = _mm_mul_ps(
      _mm_loadl_pi(_mm_setzero_ps(), (__m64*)a),
      _mm_permutevar_ps(
          _mm_loadl_pi(_mm_setzero_ps(), (__m64*)b), _mm_set_epi32(0, 0, 0, 1)
      )
  );
  return _mm_cvtss_f32(
      _mm_sub_ps(x0, _mm_permutevar_ps(x0, _mm_set_epi32(4, 3, 0, 1)))
  );
}
T_SCOPE void vec2_div(vec2 a, vec2 b, vec2 r) {
  _mm_storel_pi(
      (__m64*)r, _mm_div_ps(
                     _mm_loadl_pi(_mm_setzero_ps(), (__m64*)a),
                     _mm_loadl_pi(_mm_setzero_ps(), (__m64*)b)
                 )
  );
}
T_SCOPE float vec2_mag(vec2 a) {
  return fabsf(vec2_determinent(a, a)); // sqrt(a^2 + b^2)
}

T_SCOPE void vec3_add(vec3 a, vec3 b, vec3 r) {
  _mm_maskstore_ps(
      r, _mm_set_epi32(0, -1, -1, -1),
      _mm_add_ps(
          _mm_maskload_ps(a, _mm_set_epi32(0, -1, -1, -1)),
          _mm_maskload_ps(b, _mm_set_epi32(0, -1, -1, -1))
      )
  );
}
T_SCOPE void vec3_sub(vec3 a, vec3 b, vec3 r) {
  _mm_maskstore_ps(
      r, _mm_set_epi32(0, -1, -1, -1),
      _mm_sub_ps(
          _mm_maskload_ps(a, _mm_set_epi32(0, -1, -1, -1)),
          _mm_maskload_ps(b, _mm_set_epi32(0, -1, -1, -1))
      )
  );
}
T_SCOPE void vec3_scale(vec3 a, float b, vec3 r) {
  _mm_maskstore_ps(
      r, _mm_set_epi32(0, -1, -1, -1),
      _mm_mul_ps(
          _mm_maskload_ps(a, _mm_set_epi32(0, -1, -1, -1)), _mm_set1_ps(b)
      )
  );
}
T_SCOPE float vec3_dot_mul(vec3 a, vec3 b) {
  return _mm_cvtss_f32(_mm_dp_ps(
      _mm_maskload_ps(a, _mm_set_epi32(0, -1, -1, -1)),
      _mm_maskload_ps(b, _mm_set_epi32(0, -1, -1, -1)), 0b01110001
  ));
}
T_SCOPE void vec3_cross_mul(vec3 a, vec3 b, vec3 r) {
  // a     x     bz-cy  12-21
  // b  x  y  =  cx-az  20-02
  // c     z     ay-bx  01-10
  __m128 x0, x1, x2, x3, x4;
  x0 = _mm_maskload_ps(a, _mm_set_epi32(0, -1, -1, -1)); // 0 c b a
  x1 = _mm_maskload_ps(b, _mm_set_epi32(0, -1, -1, -1)); // 0 z y x
  x2 = _mm_permutevar_ps(x0, _mm_set_epi32(3, 0, 2, 1)); // 0 a c b
  x3 = _mm_permutevar_ps(x1, _mm_set_epi32(3, 1, 0, 2)); // 0 y x z
  x4 = _mm_mul_ps(x2, x3);                               // ay cx bz
  x2 = _mm_permutevar_ps(x0, _mm_set_epi32(3, 1, 0, 2)); // 0 b a c
  x3 = _mm_permutevar_ps(x1, _mm_set_epi32(3, 0, 2, 1)); // 0 x z y
  x4 = _mm_sub_ps(x4, _mm_mul_ps(x2, x3));               // ay-bx cx-az bz-cy
  _mm_maskstore_ps(r, _mm_set_epi32(0, -1, -1, -1), x4);
}
T_SCOPE void vec3_div(vec3 a, vec3 b, vec3 r) {
  _mm_maskstore_ps(
      r, _mm_set_epi32(0, -1, -1, -1),
      _mm_div_ps(
          _mm_maskload_ps(a, _mm_set_epi32(0, -1, -1, -1)),
          _mm_maskload_ps(b, _mm_set_epi32(0, -1, -1, -1))
      )
  );
}
T_SCOPE float vec3_mag(vec3 a) {
  return sqrtf(vec3_dot_mul(a, a)); // sqrt(a^2 + b^2 + c^2)
}
T_SCOPE void vec3_normalize(vec3 a, vec3 r) {
  float mag = vec3_mag(a);
  if(mag < EPSILON) {
    _mm_maskstore_ps(r, _mm_set_epi32(0, -1, -1, -1), _mm_setzero_ps());
    return;
  }
  _mm_maskstore_ps(
      r, _mm_set_epi32(0, -1, -1, -1),
      _mm_div_ps(
          _mm_maskload_ps(a, _mm_set_epi32(0, -1, -1, -1)), _mm_set1_ps(mag)
      )
  );
}

T_SCOPE void vec4_add(vec4 a, vec4 b, vec4 r) {
  _mm_store_ps(r, _mm_add_ps(_mm_load_ps(a), _mm_load_ps(b)));
}
T_SCOPE void vec4_sub(vec4 a, vec4 b, vec4 r) {
  _mm_store_ps(r, _mm_sub_ps(_mm_load_ps(a), _mm_load_ps(b)));
}
T_SCOPE float vec4_dot_mul(vec4 a, vec4 b) {
  return _mm_cvtss_f32(_mm_dp_ps(_mm_load_ps(a), _mm_load_ps(b), 0b11110001));
}
T_SCOPE void vec4_div(vec4 a, vec4 b, vec4 r) {
  _mm_store_ps(r, _mm_div_ps(_mm_load_ps(a), _mm_load_ps(b)));
}
T_SCOPE float vec4_mag(vec4 a) {
  return sqrtf(vec4_dot_mul(a, a)); // sqrt(a^2 + b^2 + c^2 + d^2)
}
T_SCOPE void vec4_rotate(vec4 a, float angle, vec3 axis, vec4 r) {
  m4x4 m_rot = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
  m4x4_make_rotate(m_rot, axis, angle);
  m4x4_vec4_mul(m_rot, a, r);
}
T_SCOPE void vec4_translate(vec4 a, vec3 by, vec4 r) {
  m4x4 m_tran = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
  m4x4_make_translate(m_tran, by);
  m4x4_vec4_mul(m_tran, a, r);
}
T_SCOPE void vec4_scale(vec4 a, vec3 by, vec4 r) {
  m4x4 m_sca = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
  m4x4_make_scale(m_sca, by);
  m4x4_vec4_mul(m_sca, a, r);
}

T_SCOPE void m2x2_add(m2x2 a, m2x2 b, m2x2 r) {
  vec2_add(a[0], b[0], r[0]);
  vec2_add(a[1], b[1], r[1]);
}
T_SCOPE void m2x2_sub(m2x2 a, m2x2 b, m2x2 r) {
  vec2_sub(a[0], b[0], r[0]);
  vec2_sub(a[1], b[1], r[1]);
}
T_SCOPE void m2x2_mul(m2x2 a, m2x2 b, m2x2 r) {
  __m128 x0, x1, x2, x3, x4;

  // a b  x  e f  =  ae+bg af+bh
  // c d     g h     ce+dg cf+dh

  x1 = _mm_load_ps(a[0]); // d c b a | 3 2 1 0 | 128 64 32 16
  x2 = _mm_load_ps(b[0]); // h g f e | 3 2 1 0 | 128 64 32 16

  x3 = _mm_shuffle_ps(x2, x2, _MM_SHUFFLE(1, 0, 1, 0)); // f e f e
  x4 = _mm_shuffle_ps(x2, x2, _MM_SHUFFLE(3, 2, 3, 2)); // h g h g

  x0 = _mm_shuffle_ps(x1, x1, _MM_SHUFFLE(2, 2, 0, 0)); // c c a a
  x2 = _mm_shuffle_ps(x1, x1, _MM_SHUFFLE(3, 3, 1, 1)); // d d b b

  x0 = _mm_add_ps(
      _mm_mul_ps(x0, x3), _mm_mul_ps(x2, x4)
  ); // ae+bg af+bh ce+dg cf+dh

  _mm_store_ps(r[0], x0);
}
T_SCOPE void m2x2_vec2_mul(m2x2 a, vec2 b, vec2 r) {
  __m128 x0, x1, x2, x3;

  // a b  x  e  =  ae+bf
  // c d     f     ce+df

  x1 = _mm_load_ps(a[0]); // d c b a | 3 2 1 0 | 128 64 32 16
  x2 = _mm_loadl_pi(
      _mm_setzero_ps(),
      (__m64*)b
  ); // 0 0 f e | 3 2 1 0 | 128 64 32 16

  x3 = _mm_shuffle_ps(x2, x2, _MM_SHUFFLE(1, 0, 1, 0)); // f e f e
  x3 = _mm_mul_ps(x1, x3); // df ce bf ae  (fefe * dcba)
  x0 = _mm_shuffle_ps(
      x3, x3, _MM_SHUFFLE(0, 3, 0, 1)
  ); // ae df ae bf (rearrange from df ce bf ae)
  x0 = _mm_add_ps(
      x3,
      x0
  ); // (df ce bf ae) * (ae df ae bf) = [-, ce+df, -, ae+bf]
  x0 = _mm_shuffle_ps(x0, x0, _MM_SHUFFLE(0, 0, 2, 0)); // [-,  -, ce+df, ae+bf]

  _mm_storel_pi((__m64*)r, x0);
}

T_SCOPE void m2x3_add(m2x3 a, m2x3 b, m2x3 r) {
  vec3_add(a[0], b[0], r[0]);
  vec3_add(a[1], b[1], r[1]);
}
T_SCOPE void m2x3_sub(m2x3 a, m2x3 b, m2x3 r) {
  vec3_sub(a[0], b[0], r[0]);
  vec3_sub(a[1], b[1], r[1]);
}
T_SCOPE void m2x3_vec3_mul(m2x3 a, vec3 b, vec2 r) {
  // a b c  x  g  = ag+bh+ci
  // d e f     h    dg+eh+fi
  //           i

  __m128 x0, x1, x2, x3;
  x0 = _mm_maskload_ps(a[0], _mm_set_epi32(0, -1, -1, -1));     // 0 c b a
  x1 = _mm_maskload_ps(b, _mm_set_epi32(0, -1, -1, -1));        // 0 i h g
  x2 = _mm_dp_ps(x0, x1, 0b01110001);                           // ag+bh+ci
  x0 = _mm_maskload_ps(a[0] + 3, _mm_set_epi32(0, -1, -1, -1)); // 0 f e d
  x3 = _mm_dp_ps(x0, x1, 0b01110010);                           // dg+eh+fi
  _mm_maskstore_ps(
      r, _mm_set_epi32(0, 0, -1, -1), _mm_blend_ps(x2, x3, 0b0010)
  );
}

T_SCOPE void m2x4_add(m2x4 a, m2x4 b, m2x4 r) {
  _mm256_store_ps(
      r[0], _mm256_add_ps(_mm256_load_ps(a[0]), _mm256_load_ps(b[0]))
  );
}
T_SCOPE void m2x4_sub(m2x4 a, m2x4 b, m2x4 r) {
  _mm256_store_ps(
      r[0], _mm256_sub_ps(_mm256_load_ps(a[0]), _mm256_load_ps(b[0]))
  );
}
T_SCOPE void m2x4_vec4_mul(m2x4 a, vec4 b, vec2 r) {
  // a b c d x i = ai+bj+ck+dl
  // e f g h   j   ei+fj+gk+hl
  //           k
  //           l
  __m128 x0;
  x0 = _mm_load_ps(b); // l k j i
  _mm_maskstore_ps(
      r, _mm_set_epi32(0, 0, -1, -1),
      _mm_blend_ps(
          _mm_dp_ps(_mm_load_ps(a[0]), x0, 0b11110001),
          _mm_dp_ps(_mm_load_ps(a[0] + 4), x0, 0b11110010), 0b0010
      )
  );
}

T_SCOPE void m3x2_add(m3x2 a, m3x2 b, m3x2 r) {
  _mm256_maskstore_ps(
      r[0], _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1),
      _mm256_add_ps(
          _mm256_maskload_ps(
              a[0], _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1)
          ),
          _mm256_maskload_ps(
              b[0], _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1)
          )
      )
  );
}
T_SCOPE void m3x2_sub(m3x2 a, m3x2 b, m3x2 r) {
  _mm256_maskstore_ps(
      r[0], _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1),
      _mm256_sub_ps(
          _mm256_maskload_ps(
              a[0], _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1)
          ),
          _mm256_maskload_ps(
              b[0], _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1)
          )
      )
  );
}
T_SCOPE void m3x2_vec2_mul(m3x2 a, vec2 b, vec3 r) {
  // a b     x     ax+by
  // c d  x  y  =  cx+dy
  // e f           ex+fy

  __m128 x0, x1, x2, x3;
  x0 = _mm_load_ps(a[0]);                                      // d c b a
  x1 = _mm_maskload_ps(a[0] + 4, _mm_set_epi32(0, 0, -1, -1)); // 0 0 e f
  x2 = _mm_permutevar_ps(
      _mm_maskload_ps(b, _mm_set_epi32(0, 0, -1, -1)), _mm_set_epi32(1, 0, 1, 0)
  ); // y x y x
  x3 = _mm_blend_ps(
      _mm_blend_ps(
          _mm_dp_ps(x0, x2, 0b00110001), _mm_dp_ps(x0, x2, 0b11000010), 0b0010
      ),
      _mm_dp_ps(x1, x2, 0b00110100),
      0b0100
  ); // 0 ex+fy cx+dy ax+by

  _mm_maskstore_ps(r, _mm_set_epi32(0, -1, -1, -1), x3);
}

T_SCOPE void m3x3_add(m3x3 a, m3x3 b, m3x3 r) {
  vec3_add(a[0], b[0], r[0]);
  vec3_add(a[1], b[1], r[1]);
  vec3_add(a[2], b[2], r[2]);
}
T_SCOPE void m3x3_sub(m3x3 a, m3x3 b, m3x3 r) {
  vec3_sub(a[0], b[0], r[0]);
  vec3_sub(a[1], b[1], r[1]);
  vec3_sub(a[2], b[2], r[2]);
}
T_SCOPE void m3x3_vec3_mul(m3x3 a, vec3 b, vec3 r) {
  // a b c     x     ax+by+cz
  // d e f  x  y  =  dx+ey+fz
  // g h i     z     gx+hy+iz

  __m128 x0, x1, x2, x3, x4;
  x0 = _mm_maskload_ps(b, _mm_set_epi32(0, -1, -1, -1)); // 0 z y x
  x1 = _mm_dp_ps(
      _mm_maskload_ps(a[0], _mm_set_epi32(0, -1, -1, -1)), x0,
      0b01110001
  ); // ax+by+cz
  x2 = _mm_dp_ps(
      _mm_maskload_ps(a[0] + 3, _mm_set_epi32(0, -1, -1, -1)), x0,
      0b01110010
  ); // dx+ey+fz
  x3 = _mm_dp_ps(
      _mm_maskload_ps(a[0] + 6, _mm_set_epi32(0, -1, -1, -1)), x0,
      0b01110100
  ); // gx+hy+iz
  x4 = _mm_blend_ps(x1, x2, 0b0010);
  x4 = _mm_blend_ps(x4, x3, 0b0100);
  _mm_maskstore_ps(r, _mm_set_epi32(0, -1, -1, -1), x4);
}
T_SCOPE void m3x3_mul(m3x3 a, m3x3 b, m3x3 r) {
  // a b c     j k l     aj+bm+cp ak+bn+cq al+bo+cr
  // d e f  x  m n o  =  dj+em+fp dk+en+fq dl+eo+fr
  // g h i     p q r     gj+hm+ip gk+gn+iq gl+ho+ir

  // transpose b;
  m3x3 b_t;
  m3x3_transpose(b, b_t); // r o l q n k p m j

  __m128 x0, x1, x2, x3, x4, x5;
  x0 = _mm_maskload_ps(a[0], _mm_set_epi32(0, -1, -1, -1));     // 0 c b a
  x1 = _mm_maskload_ps(a[0] + 3, _mm_set_epi32(0, -1, -1, -1)); // 0 f e d
  x2 = _mm_maskload_ps(a[0] + 6, _mm_set_epi32(0, -1, -1, -1)); // 0 g h i

  x3 = _mm_maskload_ps(b_t[0], _mm_set_epi32(0, -1, -1, -1));     // 0 p m j
  x4 = _mm_maskload_ps(b_t[0] + 3, _mm_set_epi32(0, -1, -1, -1)); // 0 q n k
  x5 = _mm_maskload_ps(b_t[0] + 6, _mm_set_epi32(0, -1, -1, -1)); // 0 r o l

  _mm_store_ps(
      r[0], _mm_blend_ps(
                _mm_blend_ps(
                    _mm_blend_ps(
                        _mm_dp_ps(x0, x3, 0b01110001),
                        _mm_dp_ps(x0, x4, 0b01110010), 0b0010
                    ),
                    _mm_dp_ps(x0, x5, 0b01110100), 0b0100
                ),
                _mm_dp_ps(x1, x3, 0b01111000), 0b1000
            )
  );
  _mm_store_ps(
      r[0] + 4, _mm_blend_ps(
                    _mm_blend_ps(
                        _mm_blend_ps(
                            _mm_dp_ps(x1, x4, 0b01110001),
                            _mm_dp_ps(x1, x5, 0b01110010), 0b0010
                        ),
                        _mm_dp_ps(x2, x3, 0b01110100), 0b0100
                    ),
                    _mm_dp_ps(x2, x4, 0b01111000), 0b1000
                )
  );
  _mm_store_ss(r[0] + 8, _mm_dp_ps(x2, x5, 0b01110001));
}

T_SCOPE void m3x4_add(m3x4 a, m3x4 b, m3x4 r) {
  _mm256_store_ps(
      r[0], _mm256_add_ps(_mm256_load_ps(a[0]), _mm256_load_ps(b[0]))
  );
  _mm_store_ps(
      r[0] + 8, _mm_add_ps(_mm_load_ps(a[0] + 8), _mm_load_ps(b[0] + 8))
  );
}
T_SCOPE void m3x4_sub(m3x4 a, m3x4 b, m3x4 r) {
  _mm256_store_ps(
      r[0], _mm256_sub_ps(_mm256_load_ps(a[0]), _mm256_load_ps(b[0]))
  );
  _mm_store_ps(
      r[0] + 8, _mm_sub_ps(_mm_load_ps(a[0] + 8), _mm_load_ps(b[0] + 8))
  );
}
T_SCOPE void m3x4_vec4_mul(m3x4 a, vec4 b, vec3 r) {
  // a b c d     w     aw+bx+cy+dz
  // e f g h  x  x  =  ew+fx+gy+hz
  // i j k l     y     iw+jx+ky+lz
  //             z

  __m128 x0, x1, x2, x3, x4;
  x0 = _mm_load_ps(a[0]);     // d c b a
  x1 = _mm_load_ps(a[0] + 4); // h g f e
  x2 = _mm_load_ps(a[0] + 8); // l k j i
  x3 = _mm_load_ps(b);        // z y x w

  x4 = _mm_blend_ps(
      _mm_blend_ps(
          _mm_dp_ps(x0, x3, 0b11110001), _mm_dp_ps(x1, x3, 0b11110010), 0b0010
      ),
      _mm_dp_ps(x2, x3, 0b11110100), 0b0100
  );
  _mm_maskstore_ps(r, _mm_set_epi32(0, -1, -1, -1), x4);
}

T_SCOPE void m4x2_add(m4x2 a, m4x2 b, m4x2 r) {
  _mm256_store_ps(
      r[0], _mm256_add_ps(_mm256_load_ps(a[0]), _mm256_load_ps(b[0]))
  );
}
T_SCOPE void m4x2_sub(m4x2 a, m4x2 b, m4x2 r) {
  _mm256_store_ps(
      r[0], _mm256_sub_ps(_mm256_load_ps(a[0]), _mm256_load_ps(b[0]))
  );
}
T_SCOPE void m4x2_vec2_mul(m4x2 a, vec2 b, vec4 r) {
  // a b     x     ax+by
  // c d  x  y  =  cx+dy
  // e f           ex+fy
  // g h           gx+hy

  __m256 y0, y1, y2;
  __m128 sum;
  y0 = _mm256_load_ps(a[0]); // h g f e d c b a
  y1 = _mm256_permutevar8x32_ps(
      _mm256_maskload_ps(b, _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1)),
      _mm256_set_epi32(1, 0, 1, 0, 1, 0, 1, 0)
  ); // y x y x y x y x
  y2 = _mm256_permutevar8x32_ps(
      _mm256_mul_ps(y0, y1), _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0)
  ); // hy fy dy by gx ex cx ax
  sum = _mm_add_ps(
      _mm256_extractf128_ps(y2, 0), _mm256_extractf128_ps(y2, 1)
  ); // gx+hy ex+fy cx+dy ax+by
  _mm_store_ps(r, sum);
}

T_SCOPE void m4x3_add(m4x3 a, m4x3 b, m4x3 r) {
  _mm256_store_ps(
      r[0], _mm256_add_ps(_mm256_load_ps(a[0]), _mm256_load_ps(b[0]))
  );
  _mm_store_ps(
      r[0] + 8, _mm_add_ps(_mm_load_ps(a[0] + 8), _mm_load_ps(b[0] + 8))
  );
}
T_SCOPE void m4x3_sub(m4x3 a, m4x3 b, m4x3 r) {
  _mm256_store_ps(
      r[0], _mm256_sub_ps(_mm256_load_ps(a[0]), _mm256_load_ps(b[0]))
  );
  _mm_store_ps(
      r[0] + 8, _mm_sub_ps(_mm_load_ps(a[0] + 8), _mm_load_ps(b[0] + 8))
  );
}
T_SCOPE void m4x3_vec3_mul(m4x3 a, vec3 b, vec4 r) {
  // a b c     x   ax+by+cz
  // d e f  x  y   dx+ey+fz
  // g h i     z   gx+hy+iz
  // j k l         jx+ky+lz

  __m128 x0, x1, x2, x3, x4, x5;
  x0 = _mm_load_ps(a[0]);                                       // d c b a
  x1 = _mm_loadu_ps(a[0] + 3);                                  // g f e d
  x2 = _mm_loadu_ps(a[0] + 6);                                  // j i h g
  x3 = _mm_maskload_ps(a[0] + 9, _mm_set_epi32(0, -1, -1, -1)); // 0 l k j
  x4 = _mm_maskload_ps(b, _mm_set_epi32(0, -1, -1, -1));        // 0 z y x
  x5 = _mm_blend_ps(
      _mm_blend_ps(
          _mm_blend_ps(
              _mm_dp_ps(x0, x4, 0b01110001), _mm_dp_ps(x1, x4, 0b01110010),
              0b0010
          ),
          _mm_dp_ps(x2, x4, 0b01110100), 0b0100
      ),
      _mm_dp_ps(x3, x4, 0b01111000),
      0b1000
  ); // jx+ky+lz gx+hy+iz dx+ey+fz ax+by+cz
  _mm_store_ps(r, x5);
}

T_SCOPE void m4x4_add(m4x4 a, m4x4 b, m4x4 r) {
  _mm256_store_ps(
      r[0], _mm256_add_ps(_mm256_load_ps(a[0]), _mm256_load_ps(b[0]))
  );
  _mm256_store_ps(
      r[0] + 8,
      _mm256_add_ps(_mm256_load_ps(a[0] + 8), _mm256_load_ps(b[0] + 8))
  );
}
T_SCOPE void m4x4_sub(m4x4 a, m4x4 b, m4x4 r) {
  _mm256_store_ps(
      r[0], _mm256_sub_ps(_mm256_load_ps(a[0]), _mm256_load_ps(b[0]))
  );
  _mm256_store_ps(
      r[0] + 8,
      _mm256_sub_ps(_mm256_load_ps(a[0] + 8), _mm256_load_ps(b[0] + 8))
  );
}
T_SCOPE void m4x4_vec4_mul(m4x4 a, vec4 b, vec4 r) {
  // a b c d     w    aw+bx+cy+dz
  // e f g h  x  x  = ew+fx+gy+hz
  // i j k l     y    iw+jx+ky+lz
  // m n o p     z    mw+nx+oy+pz

  __m128 x0, x1, x2, x3, x4, x5;
  x0 = _mm_load_ps(a[0]);      // d c b a
  x1 = _mm_load_ps(a[0] + 4);  // h g f e
  x2 = _mm_load_ps(a[0] + 8);  // l k j i
  x3 = _mm_load_ps(a[0] + 12); // p o n m
  x4 = _mm_load_ps(b);         // z y x w

  x5 = _mm_blend_ps(
      _mm_blend_ps(
          _mm_blend_ps(
              _mm_dp_ps(x0, x4, 0b11110001), _mm_dp_ps(x1, x4, 0b11110010),
              0b0010
          ),
          _mm_dp_ps(x2, x4, 0b11110100), 0b0100
      ),
      _mm_dp_ps(x3, x4, 0b11111000),
      0b1000
  ); // mw+nx+oy+pz iw+jx+ky+lz ew+fx+gy+hz aw+bx+cy+dz
  _mm_store_ps(r, x5);
}
T_SCOPE void m4x4_identity(m4x4 a) {
  a[0][0] = a[1][1] = a[2][2] = a[3][3] = 1.0f;
  a[0][1] = a[0][2] = a[0][3] = a[1][0] = a[1][2] = a[1][3] = a[2][0] =
      a[2][1] = a[2][3] = a[3][0] = a[3][1] = a[3][2] = 0.0f;
}
T_SCOPE void m4x4_mul(m4x4 a, m4x4 b, m4x4 r) {
  // a b c d     1 2 3 4        a1+b5+c9+d13 a2+b6+c10+d14 a3+b7+c11+d15
  // a4+b8+c12+d16 e f g h  x  5 6 7 8     =  e1+f5+g9+h13 e2+f6+g10+h14
  // e3+f7+g11+h15 e4+f8+g12+h16 i j k l     9 10 11 12     i1+j5+k9+l13
  // i2+j6+k10+l14 i3+j7+k11+l15 i4+j8+k12+l16 m n o p     13 14 15 16
  // m1+n5+o9+p13 m2+n6+o10+p14 m3+n7+o11+p15 m4+n8+o12+p16

  __m128 x0, x1, x2, x3, x4, x5;
  m4x4 b_t;
  m4x4_transpose(b, b_t);
  x0 = _mm_load_ps(b_t[0]);      // 13 9 5 1
  x1 = _mm_load_ps(b_t[0] + 4);  // 14 10 6 2
  x2 = _mm_load_ps(b_t[0] + 8);  // 15 11 7 3
  x3 = _mm_load_ps(b_t[0] + 12); // 16 12 8 4

  for(int i = 0; i < 4; i++) {
    x4 = _mm_load_ps(a[i]);
    x5 = _mm_blend_ps(
        _mm_blend_ps(
            _mm_blend_ps(
                _mm_dp_ps(x4, x0, 0b11110001), _mm_dp_ps(x4, x1, 0b11110010),
                0b0010
            ),
            _mm_dp_ps(x4, x2, 0b11110100), 0b0100
        ),
        _mm_dp_ps(x4, x3, 0b11111000), 0b1000
    );
    _mm_store_ps(r[i], x5);
  }
}
T_SCOPE void m4x4_make_rotate(m4x4 a, vec3 axis, float angle) {
  vec3 xn, vc, vs;
  float ac, as;
  m4x4 r = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 1}};

  ac = cosf(deg_to_rad(angle));
  as = sinf(deg_to_rad(angle));

  vec3_normalize(axis, xn);
  vec3_scale(xn, (1 - ac), vc);
  vec3_scale(xn, as, vs);

  vec3_scale(xn, vc[0], r[0]);
  vec3_scale(xn, vc[1], r[1]);
  vec3_scale(xn, vc[2], r[2]);

  r[0][0] += ac, r[1][0] -= vs[2], r[2][0] += vs[1];
  r[0][1] += vs[2], r[1][1] += ac, r[2][1] -= vs[0];
  r[0][2] -= vs[1], r[1][2] += vs[0], r[2][2] += ac;

  m4x4_mul(r, a, a);
}
T_SCOPE void m4x4_make_translate(m4x4 a, vec3 t) {
  /**
   * {1, 0, 0, 0}
   * {0, 1, 0, 0}
   * {0, 0, 1, 0}
   * {t[0], t[1], t[2], 1}
   */
  m4x4 r;
  m4x4_identity(r);
  r[3][0] = t[0];
  r[3][1] = t[1];
  r[3][2] = t[2];

  m4x4_mul(r, a, a);
}
T_SCOPE void m4x4_make_scale(m4x4 a, vec3 s) {
  /**
   * {s[0], 0   , 0   , 0}
   * {0   , s[1], 0   , 0}
   * {0   , 0   , s[2], 0}
   * {0   , 0   , 0   , 1}
   */
  m4x4 r;
  m4x4_identity(r);
  r[0][0] = s[0];
  r[1][1] = s[1];
  r[2][2] = s[2];
  r[3][3] = 1.0f;

  m4x4_mul(r, a, a);
}
T_SCOPE void m4x4_make_perspective(
    m4x4 a, float fov_x, float aspect_ratio, float n, float f
) {
  float s = 1.0f / tanf(deg_to_rad(fov_x) * 0.5f);
  float d = 1.0f / (n - f);
  m4x4 m_p = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

  m_p[0][0] = s / aspect_ratio;
  m_p[1][1] = s;
  m_p[2][2] = (n + f) * d;
  m_p[2][3] = -1.0f;
  m_p[3][2] = 2.0f * n * f * d;

  m4x4_mul(m_p, a, a);
}
T_SCOPE void m4x4_make_look_at(m4x4 a, vec3 position, vec3 target, vec3 up) {
  vec3 f, u, s;
  m4x4 m_l = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 1}};

  vec3_sub(target, position, f);
  vec3_normalize(f, f);
  vec3_cross_mul(f, up, s);
  vec3_normalize(s, s);
  vec3_cross_mul(s, f, u);

  m_l[0][0] = s[0], m_l[1][0] = s[1], m_l[2][0] = s[2];
  m_l[0][1] = u[0], m_l[1][1] = u[1], m_l[2][1] = u[2];
  m_l[0][2] = -f[0], m_l[1][2] = -f[1], m_l[2][2] = -f[2];

  m_l[3][0] = -vec3_dot_mul(s, position),
  m_l[3][1] = -vec3_dot_mul(u, position),
  m_l[3][2] = vec3_dot_mul(f, position);

  m4x4_mul(m_l, a, a);
}

T_SCOPE double deg_to_rad(double deg) {
  return deg * (MATH_PI / 180.0);
}
T_SCOPE double rad_to_deg(double rad) {
  return rad * (180.0 / MATH_PI);
}

#endif // TENSOR_IMPLEMENTATION
