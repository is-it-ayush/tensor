### tensor.

header only tensor math library for c. optimized for intel haswell
and amd broadwell (2013+ processors) with sse & avx instruction
extensions. (except for avx512, since my cpu doesn't support it.)

> the affine transformations such as `m4x4_make_rotate`,
> `m4x4_make_translate`, `m4x4_make_scale`, `m4x4_make_perspective`
> are written with opengl in mind i.e. column-major order so
> remember that! you'll have to transpose the matrix if you're
> using row-major order (like in vulkan, directx etc.)

### usage.

- open `tensor.h` file and find this line `#define TENSOR_IMPLEMENTATION`.
comment the line above this line says "COMMENT ME TO USE THE HEADER FILE"
- then include the header file in your project. below is an example.
```c
#define TENSOR_IMPLEMENTATION
#include "./include/tensor.h" // or whereever you put it.

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
- compile and run the program. (include the math
lib `-lm` and `-march=native` for simd intrinsics)
```sh
$ gcc -I./include -o main.o -c main.c -lm -march=native
$ gcc -I./include -o main main.o -lm -march=native
$ ./main
```

### test.

all functions are tested. to run the tests, do the following:

> asumming you've already cloned this repo.

```sh
$ make clean & make && ./test && make clean
```

### thoughts.

this is an extremely beta lib. i'm always updating it and
adding new functions as i need them. if you find any bugs or
have any suggestions, feel free to open an issue or a pull
request. i'm always open to feedback.

the largest function that i couldn't optimize is `m4x4_inverse`.
the function is a pain in the ass. so i'll prolly optimize it
but when i run into perf issues where the function is called
extensively.

### license.

[MIT](./LICENSE.md)
