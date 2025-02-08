#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>  // For log() function

#define WIDTH 2560
#define HEIGHT 1440
#define MAX_ITER 65535
#define COLOR_DEPTH 65535  // 16-bit max value

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

// Fix coordinate mapping for full image rendering
extern "C" __global__ void compute_mandelbrot(uint16_t *pixels) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= WIDTH || py >= HEIGHT) return;

    // Corrected coordinate mapping
    float x0 = ((float)px / (float)(WIDTH - 1)) * 3.5f - 2.5f;  
    float y0 = ((float)py / (float)(HEIGHT - 1)) * 2.0f - 1.0f; 

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

// Ensure CUDA execution completes before copying data
extern "C" void launch_mandelbrot(uint16_t *pixels) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

    compute_mandelbrot<<<numBlocks, threadsPerBlock>>>(pixels);
    cudaDeviceSynchronize();
}
