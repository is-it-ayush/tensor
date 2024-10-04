#include "../libmave.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

#define EPSILON 1e-6

void vec2_tests() {
  vec2 d0 = {1.0f, 2.0f};
  vec2 r0;

  vec2_add(d0, d0, r0);
  assert(r0[0] == 2.0f);
  assert(r0[1] == 4.0f);

  vec2_sub(d0, d0, r0);
  assert(r0[0] == 0.0f);
  assert(r0[1] == 0.0f);

  vec2_dot_mul(d0, d0, r0);
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

  vec3_dot_mul(d0, d0, r0);
  assert(r0[0] == 1.0f);
  assert(r0[1] == 4.0f);
  assert(r0[2] == 9.0f);

  vec3_cross_mul(d0, d0, r0);
  assert(r0[0] == 0.0f);
  assert(r0[1] == 0.0f);
  assert(r0[2] == 0.0f);

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

  vec4_dot_mul(d0, d0, r0);
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

void m2x2_tests() {
  m2x2 d0 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
  vec2 d1 = {1.0f, 2.0f};
  m2x2 r0;
  vec2 r1;

  m2x2_add(d0, d0, r0);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      assert(r0[i][j] == 2.0f * d0[i][j]);
    }
  }

  m2x2_sub(d0, d0, r0);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      assert(r0[i][j] == 0.0f);
    }
  }

  m2x2_mul(d0, d0, r0);
  assert(r0[0][0] == 7.0f);
  assert(r0[0][1] == 10.0f);
  assert(r0[1][0] == 15.0f);
  assert(r0[1][1] == 22.0f);

  m2x2_vec2_mul(d0, d1, r1);
  assert(r1[0] == 5.0f);
  assert(r1[1] == 11.0f);

  printf("m2x2 tests passed\n");
}
void m2x3_tests() {
  m2x3 d1 = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
  vec3 d2 = {1.0f, 2.0f, 3.0f};
  m2x3 r0;
  vec2 r1;

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
void m2x4_tests() {
  m2x4 d0 = {{1, 2, 3, 4}, {5, 6, 7, 8}};
  vec4 d1 = {1, 2, 3, 4};
  m2x4 r0;
  vec2 r1;

  m2x4_add(d0, d0, r0);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 4; j++) {
      assert(r0[i][j] == 2.0f * d0[i][j]);
    }
  }

  m2x4_sub(d0, d0, r0);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 4; j++) {
      assert(r0[i][j] == 0.0f);
    }
  }

  m2x4_vec4_mul(d0, d1, r1);
  assert(r1[0] == 30.0f);
  assert(r1[1] == 70.0f);

  printf("m2x4 tests passed\n");
}
void m3x2_tests() {
  m3x2 d0 = {{1, 2}, {3, 4}, {5, 6}};
  vec2 d1 = {1, 2};
  m3x2 r0;
  vec3 r1;

  m3x2_add(d0, d0, r0);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      assert(r0[i][j] == 2.0f * d0[i][j]);
    }
  }

  m3x2_sub(d0, d0, r0);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      assert(r0[i][j] == 0.0f);
    }
  }

  m3x2_vec2_mul(d0, d1, r1);
  assert(r1[0] == 5.0f);
  assert(r1[1] == 11.0f);
  assert(r1[2] == 17.0f);

  printf("m3x2 tests passed\n");
}
void m3x3_tests() {
  m3x3 d0 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  vec3 d1 = {1, 2, 3};
  m3x3 r0;
  vec3 r1;

  m3x3_add(d0, d0, r0);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      assert(r0[i][j] == 2.0f * d0[i][j]);
    }
  }

  m3x3_sub(d0, d0, r0);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      assert(r0[i][j] == 0.0f);
    }
  }

  m3x3_vec3_mul(d0, d1, r1);
  assert(r1[0] == 14.0f);
  assert(r1[1] == 32.0f);
  assert(r1[2] == 50.0f);

  m3x3_mul(d0, d0, r0);
  assert(r0[0][0] == 30.0f);
  assert(r0[0][1] == 36.0f);
  assert(r0[0][2] == 42.0f);
  assert(r0[1][0] == 66.0f);
  assert(r0[1][1] == 81.0f);
  assert(r0[1][2] == 96.0f);
  assert(r0[2][0] == 102.0f);
  assert(r0[2][1] == 126.0f);
  assert(r0[2][2] == 150.0f);

  printf("m3x3 tests passed\n");
}
void m3x4_tests() {
  m3x4 d0 = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
  vec4 d1 = {1, 2, 3, 4};
  m3x4 r0;
  vec3 r1;

  m3x4_add(d0, d0, r0);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      assert(r0[i][j] == 2.0f * d0[i][j]);
    }
  }

  m3x4_sub(d0, d0, r0);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      assert(r0[i][j] == 0.0f);
    }
  }

  m3x4_vec4_mul(d0, d1, r1);
  assert(r1[0] == 30.0f);
  assert(r1[1] == 70.0f);
  assert(r1[2] == 110.0f);

  printf("m3x4 tests passed\n");
}
void m4x2_tests() {
  m4x2 d0 = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
  vec2 d1 = {1, 2};
  m4x2 r0;
  vec4 r1;

  m4x2_add(d0, d0, r0);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 2; j++) {
      assert(r0[i][j] == 2.0f * d0[i][j]);
    }
  }

  m4x2_sub(d0, d0, r0);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 2; j++) {
      assert(r0[i][j] == 0.0f);
    }
  }

  m4x2_vec2_mul(d0, d1, r1);
  assert(r1[0] == 5.0f);
  assert(r1[1] == 11.0f);
  assert(r1[2] == 17.0f);
  assert(r1[3] == 23.0f);

  printf("m4x2 tests passed\n");
}
void m4x3_tests() {
  m4x3 d0 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
  vec3 d1 = {1, 2, 3};
  m4x3 r0;
  vec4 r1;

  m4x3_add(d0, d0, r0);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 3; j++) {
      assert(r0[i][j] == 2.0f * d0[i][j]);
    }
  }

  m4x3_sub(d0, d0, r0);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 3; j++) {
      assert(r0[i][j] == 0.0f);
    }
  }

  m4x3_vec3_mul(d0, d1, r1);
  assert(r1[0] == 14.0f);
  assert(r1[1] == 32.0f);
  assert(r1[2] == 50.0f);
  assert(r1[3] == 68.0f);

  printf("m4x3 tests passed\n");
}
void m4x4_tests() {
  m4x4 d0 = {{1,2,3,4}, {5,6,7,8}, {9,10,11,12}, {13,14,15,16}};
  vec4 d1 = {1, 2, 3, 4};
  m4x4 d2 = {{1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,0,1}};
  m4x4 r0;
  vec4 r1;

  m4x4_add(d0, d0, r0);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      assert(r0[i][j] == 2.0f * d0[i][j]);
    }
  }

  m4x4_sub(d0, d0, r0);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      assert(r0[i][j] == 0.0f);
    }
  }

  m4x4_vec4_mul(d0, d1, r1);
  assert(r1[0] == 30.0f);
  assert(r1[1] == 70.0f);
  assert(r1[2] == 110.0f);
  assert(r1[3] == 150.0f);

  m4x4_mul(d0, d0, r0);
  assert(r0[0][0] == 90.0f);
  assert(r0[0][1] == 100.0f);
  assert(r0[0][2] == 110.0f);
  assert(r0[0][3] == 120.0f);
  assert(r0[1][0] == 202.0f);
  assert(r0[1][1] == 228.0f);
  assert(r0[1][2] == 254.0f);
  assert(r0[1][3] == 280.0f);
  assert(r0[2][0] == 314.0f);
  assert(r0[2][1] == 356.0f);
  assert(r0[2][2] == 398.0f);
  assert(r0[2][3] == 440.0f);
  assert(r0[3][0] == 426.0f);
  assert(r0[3][1] == 484.0f);
  assert(r0[3][2] == 542.0f);
  assert(r0[3][3] == 600.0f);

  m4x4_rotate(d2, deg_to_rad(90.0), (vec3){0, 0, 1});
  assert(fabs(d2[0][0] - 1.0f) < EPSILON);
  assert(fabs(d2[0][1] - -1.0f) < EPSILON);
  assert(fabs(d2[1][0] - 1.0f) < EPSILON);
  assert(fabs(d2[1][1] - 1.0f) < EPSILON);
  assert(fabs(d2[2][2] - 1.0f) < EPSILON);
  assert(fabs(d2[3][3] - 1.0f) < EPSILON);

  printf("m4x4 tests passed\n");
}

void misc_tests() {
  double rad = deg_to_rad(180.0f);
  double deg = rad_to_deg(rad);
  assert(rad == MATH_PI);
  assert(deg == 180.0f);
  printf("misc tests passed\n");
}

int main() {
  vec2_tests();
  vec3_tests();
  vec4_tests();

  m2x2_tests();
  m2x3_tests();
  m2x4_tests();
  m3x2_tests();
  m3x3_tests();
  m3x4_tests();
  m4x2_tests();
  m4x3_tests();
  m4x4_tests();

  misc_tests();
  return 0;
}
