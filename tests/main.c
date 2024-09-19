#include "../libmave.h"
#include <assert.h>
#include <stdio.h>

void vec2_tests() {
  vec2 d0 = {1.0f, 2.0f};
  vec2 r0;

  vec2_add(d0, d0, r0);
  assert(r0[0] == 2.0f);
  assert(r0[1] == 4.0f);

  vec2_sub(d0, d0, r0);
  assert(r0[0] == 0.0f);
  assert(r0[1] == 0.0f);

  vec2_mul(d0, d0, r0);
  assert(r0[0] == 1.0f);
  assert(r0[1] == 4.0f);

  vec2_div(d0, d0, r0);
  assert(r0[0] == 1.0f);
  assert(r0[1] == 1.0f);

  printf("vec2 tests passed\n");
}
void vec3_tests() {
  vec3 d0 = {1.0f, 2.0f, 3.0f};
  vec3 r0;

  vec3_add(d0, d0, r0);
  assert(r0[0] == 2.0f);
  assert(r0[1] == 4.0f);
  assert(r0[2] == 6.0f);

  vec3_sub(d0, d0, r0);
  assert(r0[0] == 0.0f);
  assert(r0[1] == 0.0f);
  assert(r0[2] == 0.0f);

  vec3_mul(d0, d0, r0);
  assert(r0[0] == 1.0f);
  assert(r0[1] == 4.0f);
  assert(r0[2] == 9.0f);

  vec3_div(d0, d0, r0);
  assert(r0[0] == 1.0f);
  assert(r0[1] == 1.0f);
  assert(r0[2] == 1.0f);

  printf("vec3 tests passed\n");
}
void vec4_tests() {
  vec4 d0 = {1.0f, 2.0f, 3.0f, 4.0f};
  vec4 r0;

  vec4_add(d0, d0, r0);
  assert(r0[0] == 2.0f);
  assert(r0[1] == 4.0f);
  assert(r0[2] == 6.0f);
  assert(r0[3] == 8.0f);

  vec4_sub(d0, d0, r0);
  assert(r0[0] == 0.0f);
  assert(r0[1] == 0.0f);
  assert(r0[2] == 0.0f);
  assert(r0[3] == 0.0f);

  vec4_mul(d0, d0, r0);
  assert(r0[0] == 1.0f);
  assert(r0[1] == 4.0f);
  assert(r0[2] == 9.0f);
  assert(r0[3] == 16.0f);

  vec4_div(d0, d0, r0);
  assert(r0[0] == 1.0f);
  assert(r0[1] == 1.0f);
  assert(r0[2] == 1.0f);
  assert(r0[3] == 1.0f);

  printf("vec4 tests passed\n");
}
void vec_to_vec_tests() {
  vec2_tests();
  vec3_tests();
  vec4_tests();
}

void m2x2_tests() {
  m2x2 d0 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
  vec2 d1 = {1.0f, 2.0f};
  m2x3 d2 = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
  m2x4 d3 = {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}};
  m3x3 d4 = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}};
  m2x2 r0;
  vec2 r1;
  m2x3 r2;
  m2x4 r3;

  m2x2_add(d0, d0, r0);
  assert(r0[0][0] == 2.0f);
  assert(r0[0][1] == 4.0f);
  assert(r0[1][0] == 6.0f);
  assert(r0[1][1] == 8.0f);

  m2x2_sub(d0, d0, r0);
  assert(r0[0][0] == 0.0f);
  assert(r0[0][1] == 0.0f);
  assert(r0[1][0] == 0.0f);
  assert(r0[1][1] == 0.0f);

  m2x2_mul(d0, d0, r0);
  assert(r0[0][0] == 7.0f);
  assert(r0[0][1] == 10.0f);
  assert(r0[1][0] == 15.0f);
  assert(r0[1][1] == 22.0f);

  m2x2_vec2_mul(d0, d1, r1);
  assert(r1[0] == 5.0f);
  assert(r1[1] == 11.0f);

  m2x2_m2x3_mul(d0, d2, r2);
  assert(r2[0][0] == 9.0f);
  assert(r2[0][1] == 12.0f);
  assert(r2[0][2] == 15.0f);
  assert(r2[1][0] == 19.0f);
  assert(r2[1][1] == 26.0f);
  assert(r2[1][2] == 33.0f);

  m2x2_m2x4_mul(d0, d3, r3);
  assert(r3[0][0] == 11.0f);
  assert(r3[0][1] == 14.0f);
  assert(r3[0][2] == 17.0f);
  assert(r3[0][3] == 20.0f);
  assert(r3[1][0] == 23.0f);
  assert(r3[1][1] == 30.0f);
  assert(r3[1][2] == 37.0f);
  assert(r3[1][3] == 44.0f);

  printf("m2x2 tests passed\n");
}

void m2x3_tests() {
  m2x3 d1 = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
  vec3 d2 = {1.0f, 2.0f, 3.0f};
  m3x2 d3 = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
  m2x3 r0;
  vec2 r1;
  m2x2 r2;

  m2x3_add(d1, d1, r0);
  assert(r0[0][0] == 2.0f);
  assert(r0[0][1] == 4.0f);
  assert(r0[0][2] == 6.0f);
  assert(r0[1][0] == 8.0f);
  assert(r0[1][1] == 10.0f);
  assert(r0[1][2] == 12.0f);

  m2x3_sub(d1, d1, r0);
  assert(r0[0][0] == 0.0f);
  assert(r0[0][1] == 0.0f);
  assert(r0[0][2] == 0.0f);
  assert(r0[1][0] == 0.0f);
  assert(r0[1][1] == 0.0f);
  assert(r0[1][2] == 0.0f);

  m2x3_vec3_mul(d1, d2, r1);
  assert(r1[0] == 14.0f);
  assert(r1[1] == 32.0f);

  printf("m2x3 tests passed\n");
}

void mtx_to_mtx_tests() {
  m2x2_tests();
  m2x3_tests();
}

int main() {
  vec_to_vec_tests();
  mtx_to_mtx_tests();
  return 0;
}
