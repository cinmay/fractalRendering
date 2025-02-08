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

// Vibrant OLED color mapping
__device__ void get_color(uint8_t iter, uint8_t *r, uint8_t *g, uint8_t *b) {
    if (iter == MAX_ITER) {
        *r = 0; *g = 0; *b = 0;  // Mandelbrot body is now black
        return;
    }

    float t = (float)iter / MAX_ITER; // Normalize iteration count

    if (t < 0.2f) {
        // Black to deep purple transition
        *r = (uint8_t)(t * 80);
        *g = 0;
        *b = (uint8_t)(t * 255);
    } 
    else if (t < 0.5f) {
        // Purple to vibrant pink
        *r = (uint8_t)((t - 0.2f) * 255);
        *g = 0;
        *b = 255;
    } 
    else if (t < 0.8f) {
        // Pink to orange transition
        *r = 255;
        *g = (uint8_t)((t - 0.5f) * 255);
        *b = (uint8_t)((0.8f - t) * 255);
    } 
    else {
        // Orange fading into black
        *r = (uint8_t)((1.0f - t) * 255);
        *g = (uint8_t)((1.0f - t) * 80);
        *b = 0;
    }
}

// Compute Mandelbrot fractal with new color mapping
extern "C" __global__ void compute_mandelbrot(uint8_t *pixels) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= WIDTH || py >= HEIGHT) return;

    float x0 = ((float)px / WIDTH) * 3.5f - 2.5f;
    float y0 = ((float)py / HEIGHT) * 2.0f - 1.0f;

    uint8_t iter = mandelbrot(x0, y0);

    uint8_t r, g, b;
    get_color(iter, &r, &g, &b);

    int index = (py * WIDTH + px) * 3;
    pixels[index] = r;
    pixels[index + 1] = g;
    pixels[index + 2] = b;
}

// Ensure CUDA execution completes before copying data
extern "C" void launch_mandelbrot(uint8_t *pixels) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(WIDTH / threadsPerBlock.x, HEIGHT / threadsPerBlock.y);

    compute_mandelbrot<<<numBlocks, threadsPerBlock>>>(pixels);
    cudaDeviceSynchronize();
}
