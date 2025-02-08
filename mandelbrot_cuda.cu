#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#define WIDTH 2560
#define HEIGHT 1440
#define MAX_ITER 65535
#define COLOR_DEPTH 65535  

// Global variables for position & zoom (updated via Zig)
__device__ float offset_x = -0.5f;
__device__ float offset_y = 0.0f;
__device__ float zoom = 1.0f;

__device__ uint16_t mandelbrot(float x, float y) {
    float zx = 0.0, zy = 0.0;
    uint16_t iter = 0;

    while (zx * zx + zy * zy < 4.0f && iter < MAX_ITER) {
        float temp = zx * zx - zy * zy + x;
        zy = 2.0f * zx * zy + y;
        zx = temp;
        iter++;
    }
    return iter;
}

extern "C" __global__ void compute_mandelbrot(uint16_t *pixels, float new_offset_x, float new_offset_y, float new_zoom) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= WIDTH || py >= HEIGHT) return;

    // Update global offsets dynamically
    offset_x = new_offset_x;
    offset_y = new_offset_y;
    zoom = new_zoom;

    // Compute scaled coordinates based on offset & zoom
    float x0 = (px - WIDTH / 2.0f) / (WIDTH / 2.0f) * (3.5f / zoom) + offset_x;
    float y0 = (py - HEIGHT / 2.0f) / (HEIGHT / 2.0f) * (2.0f / zoom) + offset_y;

    uint16_t iter = mandelbrot(x0, y0);

    // Nonlinear color mapping
    float t = logf((float)iter + 1) / logf((float)MAX_ITER);
    uint16_t r = (uint16_t)(t * COLOR_DEPTH);
    uint16_t g = (uint16_t)((t * 0.8f) * COLOR_DEPTH);
    uint16_t b = (uint16_t)((t * 0.5f) * COLOR_DEPTH);

    int index = (py * WIDTH + px) * 3;
    pixels[index] = r;
    pixels[index + 1] = g;
    pixels[index + 2] = b;
}

extern "C" void launch_mandelbrot(uint16_t *pixels, float offset_x, float offset_y, float zoom) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

    compute_mandelbrot<<<numBlocks, threadsPerBlock>>>(pixels, offset_x, offset_y, zoom);
    cudaDeviceSynchronize();
}
