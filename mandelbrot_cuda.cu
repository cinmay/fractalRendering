#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#define WIDTH 2560
#define HEIGHT 1440
#define MAX_ITER 255

__device__ uint8_t mandelbrot(float x, float y) {
    float zx = 0.0, zy = 0.0;
    uint8_t iter = 0;

    while (zx * zx + zy * zy < 4.0f && iter < MAX_ITER) {
        float temp = zx * zx - zy * zy + x;
        zy = 2.0f * zx * zy + y;
        zx = temp;
        iter++;
    }
    return iter;
}

// Ensure CUDA function is correctly linked to Zig
extern "C" __global__ void compute_mandelbrot(uint8_t *pixels) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= WIDTH || py >= HEIGHT) return;

    float x0 = ((float)px / WIDTH) * 3.5f - 2.5f;
    float y0 = ((float)py / HEIGHT) * 2.0f - 1.0f;

    uint8_t iter = mandelbrot(x0, y0);

    int index = (py * WIDTH + px) * 3;
    pixels[index] = iter * 9 % 256;     // Red channel
    pixels[index + 1] = iter * 15 % 256; // Green channel
    pixels[index + 2] = iter * 5 % 256;  // Blue channel
}

// Ensure CUDA execution completes before copying data
extern "C" void launch_mandelbrot(uint8_t *pixels) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(WIDTH / threadsPerBlock.x, HEIGHT / threadsPerBlock.y);

    compute_mandelbrot<<<numBlocks, threadsPerBlock>>>(pixels);
    cudaDeviceSynchronize(); // Make sure the kernel has finished before returning
}
