# libmave.

stupid vector matrix library for c and c++ optimized
for intel haswell and amd broadwell (2013+ processors)
with sse & avx instruction extensions. (avx512)

```
# addition operations

# multiplication operations
2x2 * 2x1 (m2x2_vec2_mul) | 2x2 * 2x2 (m2x2_mul) | -                                | -
2x3 * 3x1 (m2x3_vec3_mul) |  -                   | -                                | -
2x4 * 4x1 (m2x4_vec4_mul) |  -                   | -                                | -
3x2 * 2x1 (m3x2_vec2_mul) |  -                   | -                                | -
3x3 * 3x1 (m3x3_vec3_mul) |  -                   | 3x3 * 3x3 (m3x3_m3x3_mul)        | -
3x4 * 4x1 (m3x4_vec4_mul) |  -                   | -                                | -
4x2 * 2x1 (m4x2_vec2_mul) |  -                   | -                                | -
4x3 * 3x1 (m4x2_vec3_mul) |  -                   | -                                | -
4x4 * 4x1 (m4x4_vec4_mul) |  -                   | -                                | 4x4 * 4x4 (m4x4_m4x4_mul)
```
