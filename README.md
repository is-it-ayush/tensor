# libmave.

stupid vector matrix library for c and c++ optimized
for intel haswell and amd broadwell (2013+ processors)
with sse & avx instruction extensions. (avx512)

# operations.

```
# addition operations
2x1 + 2x1 (vec2_add)      | 2x2 + 2x2 (m2x2_add) | 2x3 + 2x3 (m2x3_add) | 2x4 + 2x4 (m2x4_add)
3x1 + 3x1 (vec3_add)      | 3x2 + 3x2 (m3x2_add) | 3x3 + 3x3 (m3x3_add) | 3x4 + 3x4 (m3x4_add)
4x1 + 4x1 (vec4_add)      | 4x2 + 4x2 (m4x2_add) | 4x3 + 4x3 (m4x3_add) | 4x4 + 4x4 (m4x4_add)

# subtraction operations
2x1 - 2x1 (vec2_sub)      | 2x2 - 2x2 (m2x2_sub) | 2x3 - 2x3 (m2x3_sub) | 2x4 - 2x4 (m2x4_sub)
3x1 - 3x1 (vec3_sub)      | 3x2 - 3x2 (m3x2_sub) | 3x3 - 3x3 (m3x3_sub) | 3x4 - 3x4 (m3x4_sub)
4x1 - 4x1 (vec4_sub)      | 4x2 - 4x2 (m4x2_sub) | 4x3 - 4x3 (m4x3_sub) | 4x4 - 4x4 (m4x4_sub)

# division operations
2x1 / 2x1 (vec2_div)      | -                    | -                    | -
3x1 / 3x1 (vec3_div)      | -                    | -                    | -
4x1 / 4x1 (vec4_div)      | -                    | -                    | -

# multiplication operations
2x1 * 2x1 (vec2_dot_mul)      | -                          | -                    | -
3x1 * 3x1 (vec3_dot_mul)      | 3x1 x 3x1 (vec3_cross_mul) | -                    | -
4x1 * 4x1 (vec4_dot_mul)      | -                          | -                    | -
2x2 * 2x1 (m2x2_vec2_mul)     | 2x2 * 2x2 (m2x2_mul)       | -                    | -
2x3 * 3x1 (m2x3_vec3_mul)     |  -                         | -                    | -
2x4 * 4x1 (m2x4_vec4_mul)     |  -                         | -                    | -
3x2 * 2x1 (m3x2_vec2_mul)     |  -                         | -                    | -
3x3 * 3x1 (m3x3_vec3_mul)     |  -                         | 3x3 * 3x3 (m3x3_mul) | -
3x4 * 4x1 (m3x4_vec4_mul)     |  -                         | -                    | -
4x2 * 2x1 (m4x2_vec2_mul)     |  -                         | -                    | -
4x3 * 3x1 (m4x2_vec3_mul)     |  -                         | -                    | -
4x4 * 4x1 (m4x4_vec4_mul)     |  -                         | -                    | 4x4 * 4x4 (m4x4_mul)

# transpose operations
m2x2 (m2x2_transpose)     |  -                   | -                    | -
m3x3 (m3x3_transpose)     |  -                   | -                    | -
m4x4 (m4x4_transpose)     |  -                   | -                    | -
```
