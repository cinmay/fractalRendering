const std = @import("std");
const sdl = @cImport(@cInclude("SDL2/SDL.h"));

const width = 2560;
const height = 1440;

// Declare CUDA functions
extern fn cudaMalloc(ptr: *[*]u16, size: usize) c_int;
extern fn cudaFree(ptr: [*]u16) c_int;
extern fn cudaMemcpy(dst: [*]u16, src: [*]const u16, size: usize, kind: c_int) c_int;
extern fn launch_mandelbrot(pixels: [*]u16, offset_x: f32, offset_y: f32, zoom: f32) void;

// Mandelbrot navigation variables
var offset_x: f32 = -0.5;
var offset_y: f32 = 0.0;
var zoom: f32 = 1.0;
const move_step: f32 = 0.1;
const zoom_factor: f32 = 1.2;

fn render_mandelbrot(pixels: []u16, cuda_pixels: [*]u16) !void {
    launch_mandelbrot(cuda_pixels, offset_x, offset_y, zoom);

    const cudaMemcpyDeviceToHost = 2;
    if (cudaMemcpy(pixels.ptr, cuda_pixels, width * height * 3 * 2, cudaMemcpyDeviceToHost) != 0) {
        std.debug.print("❌ CUDA memcpy failed! Memory copy issue.\n", .{});
        return;
    }
}

pub fn main() !void {
    if (sdl.SDL_Init(sdl.SDL_INIT_VIDEO) != 0) {
        std.debug.print("Failed to initialize SDL2: {s}\n", .{sdl.SDL_GetError()});
        return;
    }
    defer sdl.SDL_Quit();

    const window = sdl.SDL_CreateWindow("CUDA Mandelbrot Navigation", sdl.SDL_WINDOWPOS_CENTERED, sdl.SDL_WINDOWPOS_CENTERED, width, height, sdl.SDL_WINDOW_SHOWN);
    if (window == null) return;
    defer sdl.SDL_DestroyWindow(window);

    const renderer = sdl.SDL_CreateRenderer(window, -1, sdl.SDL_RENDERER_ACCELERATED);
    if (renderer == null) return;
    defer sdl.SDL_DestroyRenderer(renderer);

    const pixels_size = width * height * 3 * 2;
    const pixels = try std.heap.page_allocator.alloc(u16, pixels_size / 2);
    defer std.heap.page_allocator.free(pixels);

    var pixels_8bit = try std.heap.page_allocator.alloc(u8, pixels_size / 2);
    defer std.heap.page_allocator.free(pixels_8bit);

    var cuda_pixels: [*]u16 = undefined;
    if (cudaMalloc(&cuda_pixels, pixels_size) != 0) {
        std.debug.print("❌ CUDA malloc failed!\n", .{});
        return;
    }
    defer _ = cudaFree(cuda_pixels);

    try render_mandelbrot(pixels, cuda_pixels);

    // Convert 16-bit pixels to 8-bit for SDL rendering
    for (0..(width * height * 3)) |i| {
        pixels_8bit[i] = @truncate(pixels[i] >> 8);
    }

    const texture = sdl.SDL_CreateTexture(renderer, sdl.SDL_PIXELFORMAT_RGB24, sdl.SDL_TEXTUREACCESS_STREAMING, width, height);
    if (texture == null) return;
    defer sdl.SDL_DestroyTexture(texture);

    _ = sdl.SDL_UpdateTexture(texture, null, pixels_8bit.ptr, width * 3);

    var running = true;
    var event: sdl.SDL_Event = undefined;

    while (running) {
        while (sdl.SDL_PollEvent(&event) != 0) {
            switch (event.type) {
                sdl.SDL_QUIT => running = false,
                sdl.SDL_KEYDOWN => {
                    switch (event.key.keysym.sym) {
                        sdl.SDLK_q, sdl.SDLK_ESCAPE => running = false,
                        sdl.SDLK_a => offset_x -= move_step / zoom, // Move left
                        sdl.SDLK_s => offset_x += move_step / zoom, // Move right
                        sdl.SDLK_w => offset_y -= move_step / zoom, // Move up
                        sdl.SDLK_r => offset_y += move_step / zoom, // Move down
                        sdl.SDLK_t => zoom *= zoom_factor, // Zoom in
                        sdl.SDLK_d => zoom /= zoom_factor, // Zoom out
                        else => {},
                    }
                    try render_mandelbrot(pixels, cuda_pixels);
                    for (0..(width * height * 3)) |i| {
                        pixels_8bit[i] = @truncate(pixels[i] >> 8);
                    }
                    _ = sdl.SDL_UpdateTexture(texture, null, pixels_8bit.ptr, width * 3);
                },
                else => {},
            }
        }
        _ = sdl.SDL_RenderClear(renderer);
        _ = sdl.SDL_RenderCopy(renderer, texture, null, null);
        sdl.SDL_RenderPresent(renderer);
    }
}
