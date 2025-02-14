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

// New double precision Mandelbrot iteration function.
__device__ uint16_t mandelbrot_dp(double x, double y) {
    double zx = 0.0, zy = 0.0;
    uint16_t iter = 0;
    while (zx * zx + zy * zy < 4.0 && iter < MAX_ITER) {
        double temp = zx * zx - zy * zy + x;
        zy = 2.0 * zx * zy + y;
        zx = temp;
        iter++;
    }
    return iter;
}

extern "C" __global__ void compute_mandelbrot(uint16_t *pixels, float new_offset_x, float new_offset_y, float new_zoom) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= WIDTH || py >= HEIGHT) return;

    // Use double precision for better accuracy on high zoom levels.
    double offset_x_d = (double)new_offset_x;
    double offset_y_d = (double)new_offset_y;
    double zoom_d = (double)new_zoom;

    // Convert pixel coordinates to normalized device coordinates [-1, 1].
    double u = ((double)px - (double)WIDTH / 2.0) / ((double)WIDTH / 2.0);
    double v = ((double)py - (double)HEIGHT / 2.0) / ((double)HEIGHT / 2.0);

    // Compute the unrotated complex coordinates using double precision.
    double x_unrot = u * (3.5 / zoom_d) + offset_x_d;
    double y_unrot = v * (2.0 / zoom_d) + offset_y_d;

    // Apply a 90° clockwise rotation: (x, y) -> (y, -x)
    double x0 = y_unrot;
    double y0 = -x_unrot;

    // Compute the iteration count using double precision.
    uint16_t iter = mandelbrot_dp(x0, y0);

    // Normalize iteration count into [0,1] using logarithmic scaling.
    float t = logf((float)iter + 1.0f) / logf((float)MAX_ITER);

    float r, g, b;
    // Force low iteration counts to black.
    const float low_threshold = 0.1f;

    if (t <= 0.35f) {
        float u_val = 0.0f;
        if (t > low_threshold) {
            u_val = (t - low_threshold) / (0.35f - low_threshold);
        }
        // Interpolate from black to neon pink.
        r = u_val * 255.0f;
        g = u_val * 20.0f;
        b = u_val * 147.0f;
    } else if (t <= 0.5f) {
        float u_val = (t - 0.35f) / (0.5f - 0.35f);
        r = 255.0f;
        g = 20.0f + u_val * (255.0f - 20.0f);
        b = 147.0f + u_val * (255.0f - 147.0f);
    } else if (t <= 0.65f) {
        float u_val = (t - 0.5f) / (0.65f - 0.5f);
        r = 255.0f;
        g = 255.0f + u_val * (165.0f - 255.0f);
        b = 255.0f + u_val * (0.0f - 255.0f);
    } else {
        float u_val = (t - 0.65f) / (1.0f - 0.65f);
        r = 255.0f + u_val * (0.0f - 255.0f);
        g = 165.0f + u_val * (0.0f - 165.0f);
        b = 0.0f;
    }

    // Scale colors from 0–255 to 16‑bit color depth.
    uint16_t r_scaled = (uint16_t)(r * 257.0f);
    uint16_t g_scaled = (uint16_t)(g * 257.0f);
    uint16_t b_scaled = (uint16_t)(b * 257.0f);

    int index = (py * WIDTH + px) * 3;
    pixels[index]     = r_scaled;
    pixels[index + 1] = g_scaled;
    pixels[index + 2] = b_scaled;
}

extern "C" void launch_mandelbrot(uint16_t *pixels, float offset_x, float offset_y, float zoom) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

    compute_mandelbrot<<<numBlocks, threadsPerBlock>>>(pixels, offset_x, offset_y, zoom);
    cudaDeviceSynchronize();
}
