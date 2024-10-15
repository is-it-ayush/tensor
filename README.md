### tensor.

tensor math library for c is optimized for intel haswell and
amd broadwell (2013+ processors) with sse & avx instruction
extensions. (except for avx512, since my cpu doesn't support it)

> the affine transformations such as `m4x4_make_rotate`,
> `m4x4_make_translate`, `m4x4_make_scale`, `m4x4_make_perspective`
> are written with opengl in mind i.e. column-major order so
> remember that!

### usage.

```c
#define TENSOR_IMPLEMENTATION
#include "./include/tensor.h"

int main() {
    // example usage.
    vec3 a = {1.0f, 2.0f, 3.0f};
    vec3 b = {4.0f, 5.0f, 6.0f};
    vec3 r;
    vec3_add(&r, &a, &b);
    print_vec3(&r);
    return 0;
}
```

```sh
$ gcc -I./include -o main.o -c main.c -lm -march=native
$ gcc -I./include -o main main.o -lm -march=native
$ ./main
```
