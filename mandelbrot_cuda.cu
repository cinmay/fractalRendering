#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#define WIDTH 2560
#define HEIGHT 1440
#define MAX_ITER 65535  // Fix off-by-one error
#define COLOR_DEPTH 65535  // 16-bit max value

__device__ uint16_t mandelbrot(float x, float y) {
    float zx = 0.0, zy = 0.0;
    uint16_t iter = 0; 

    while (zx * zx + zy * zy < 4.0f && iter < MAX_ITER) {
        if (iter > MAX_ITER) break;  // Prevent infinite loops
        float temp = zx * zx - zy * zy + x;
        zy = 2.0f * zx * zy + y;
        zx = temp;
        iter++;
    }
    return iter;
}

// Ensure correct memory usage
__device__ void get_color(uint16_t iter, uint16_t *r, uint16_t *g, uint16_t *b) {
    if (iter == MAX_ITER) {
        *r = 0; *g = 0; *b = 0;  // Mandelbrot body remains black
        return;
    }

    float t = (float)iter / MAX_ITER; 

    if (t < 0.2f) {
        *r = (uint16_t)(t * 8000);
        *g = 0;
        *b = (uint16_t)(t * COLOR_DEPTH);
    } 
    else if (t < 0.5f) {
        *r = (uint16_t)((t - 0.2f) * COLOR_DEPTH);
        *g = 0;
        *b = COLOR_DEPTH;
    } 
    else if (t < 0.8f) {
        *r = COLOR_DEPTH;
        *g = (uint16_t)((t - 0.5f) * COLOR_DEPTH);
        *b = (uint16_t)((0.8f - t) * COLOR_DEPTH);
    } 
    else {
        *r = (uint16_t)((1.0f - t) * COLOR_DEPTH);
        *g = (uint16_t)((1.0f - t) * 8000);
        *b = 0;
    }
}

// Fix memory allocation issue
extern "C" __global__ void compute_mandelbrot(uint16_t *pixels) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= WIDTH || py >= HEIGHT) return;

    float x0 = ((float)px / WIDTH) * 3.5f - 2.5f;
    float y0 = ((float)py / HEIGHT) * 2.0f - 1.0f;

    uint16_t iter = mandelbrot(x0, y0);

    uint16_t r, g, b;
    get_color(iter, &r, &g, &b);

    int index = (py * WIDTH + px) * 3;
    if (index < WIDTH * HEIGHT * 3) { // Prevent out-of-bounds writes
        pixels[index] = r;
        pixels[index + 1] = g;
        pixels[index + 2] = b;
    }
}

extern "C" void launch_mandelbrot(uint16_t *pixels) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((WIDTH + 31) / 32, (HEIGHT + 31) / 32);  // Reduce active blocks

    compute_mandelbrot<<<numBlocks, threadsPerBlock>>>(pixels);
    cudaDeviceSynchronize();
}
