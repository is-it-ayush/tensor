#include "../libmave.h"
#include <assert.h>
#include <stdio.h>

void vec2_tests() {
  vec2 data = {1.0f, 2.0f};
  vec2 result = {0.0f, 0.0f};

  vec2_add(data, data, result);
  assert(result[0] == 2.0f);
  assert(result[1] == 4.0f);

  vec2_sub(data, data, result);
  assert(result[0] == 0.0f);
  assert(result[1] == 0.0f);

  vec2_mul(data, data, result);
  assert(result[0] == 1.0f);
  assert(result[1] == 4.0f);

  vec2_div(data, data, result);
  assert(result[0] == 1.0f);
  assert(result[1] == 1.0f);

  printf("vec2 tests passed\n");
}
void vec3_tests() {
  vec3 data = {1.0f, 2.0f, 3.0f};
  vec3 result = {0.0f, 0.0f, 0.0f};

  vec3_add(data, data, result);
  assert(result[0] == 2.0f);
  assert(result[1] == 4.0f);
  assert(result[2] == 6.0f);

  vec3_sub(data, data, result);
  assert(result[0] == 0.0f);
  assert(result[1] == 0.0f);
  assert(result[2] == 0.0f);

  vec3_mul(data, data, result);
  assert(result[0] == 1.0f);
  assert(result[1] == 4.0f);
  assert(result[2] == 9.0f);

  vec3_div(data, data, result);
  assert(result[0] == 1.0f);
  assert(result[1] == 1.0f);
  assert(result[2] == 1.0f);

  printf("vec3 tests passed\n");
}
void vec4_tests() {
  vec4 data = {1.0f, 2.0f, 3.0f, 4.0f};
  vec4 result = {0.0f, 0.0f, 0.0f, 0.0f};

  vec4_add(data, data, result);
  assert(result[0] == 2.0f);
  assert(result[1] == 4.0f);
  assert(result[2] == 6.0f);
  assert(result[3] == 8.0f);

  vec4_sub(data, data, result);
  assert(result[0] == 0.0f);
  assert(result[1] == 0.0f);
  assert(result[2] == 0.0f);
  assert(result[3] == 0.0f);

  vec4_mul(data, data, result);
  assert(result[0] == 1.0f);
  assert(result[1] == 4.0f);
  assert(result[2] == 9.0f);
  assert(result[3] == 16.0f);

  vec4_div(data, data, result);
  assert(result[0] == 1.0f);
  assert(result[1] == 1.0f);
  assert(result[2] == 1.0f);
  assert(result[3] == 1.0f);

  printf("vec4 tests passed\n");
}
void vec_to_vec_tests() {
  vec2_tests();
  vec3_tests();
  vec4_tests();
}

int main() {
  vec_to_vec_tests();
  return 0;
}
