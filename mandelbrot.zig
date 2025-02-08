const std = @import("std");
const sdl = @cImport(@cInclude("SDL2/SDL.h"));

const width = 2560;
const height = 1440;

// Declare CUDA functions
extern fn cudaMalloc(ptr: *[*]u8, size: usize) c_int;
extern fn cudaFree(ptr: [*]u8) c_int;
extern fn cudaMemcpy(dst: [*]u8, src: [*]const u8, size: usize, kind: c_int) c_int;
extern fn launch_mandelbrot(pixels: [*]u8) void;

pub fn save_to_ppm(pixels: []u8) !void {
    var file = try std.fs.cwd().createFile("mandelbrot.ppm", .{ .truncate = true });
    defer file.close();

    try file.writer().print("P6\n{} {}\n255\n", .{ width, height });
    try file.writer().writeAll(pixels);
    std.debug.print("✅ Mandelbrot saved to mandelbrot.ppm\n", .{});
}

pub fn main() !void {
    if (sdl.SDL_Init(sdl.SDL_INIT_VIDEO) != 0) {
        std.debug.print("Failed to initialize SDL2: {s}\n", .{sdl.SDL_GetError()});
        return;
    }
    defer sdl.SDL_Quit();

    const window = sdl.SDL_CreateWindow("CUDA Mandelbrot", sdl.SDL_WINDOWPOS_CENTERED, sdl.SDL_WINDOWPOS_CENTERED, width, height, sdl.SDL_WINDOW_SHOWN);
    if (window == null) return;
    defer sdl.SDL_DestroyWindow(window);

    const renderer = sdl.SDL_CreateRenderer(window, -1, sdl.SDL_RENDERER_ACCELERATED);
    if (renderer == null) return;
    defer sdl.SDL_DestroyRenderer(renderer);

    const pixels_size = width * height * 3;
    const pixels = try std.heap.page_allocator.alloc(u8, pixels_size);
    defer std.heap.page_allocator.free(pixels);

    // Allocate CUDA memory
    var cuda_pixels: [*]u8 = undefined;
    if (cudaMalloc(&cuda_pixels, pixels_size) != 0) {
        std.debug.print("❌ CUDA malloc failed!\n", .{});
        return;
    }
    defer _ = cudaFree(cuda_pixels);

    // Run Mandelbrot computation on CUDA
    launch_mandelbrot(cuda_pixels);

    // Copy result back from CUDA to Zig
    const cudaMemcpyDeviceToHost = 2;
    if (cudaMemcpy(pixels.ptr, cuda_pixels, pixels_size, cudaMemcpyDeviceToHost) != 0) {
        std.debug.print("❌ CUDA memcpy failed!\n", .{});
        return;
    }

    // Save output image
    try save_to_ppm(pixels);

    // Display in SDL2 window
    const texture = sdl.SDL_CreateTexture(renderer, sdl.SDL_PIXELFORMAT_RGB24, sdl.SDL_TEXTUREACCESS_STREAMING, width, height);
    if (texture == null) return;
    defer sdl.SDL_DestroyTexture(texture);

    _ = sdl.SDL_UpdateTexture(texture, null, pixels.ptr, width * 3);

    var running = true;
    var event: sdl.SDL_Event = undefined;

    while (running) {
        while (sdl.SDL_PollEvent(&event) != 0) {
            switch (event.type) {
                sdl.SDL_QUIT => running = false,
                sdl.SDL_KEYDOWN => {
                    switch (event.key.keysym.sym) {
                        sdl.SDLK_q, sdl.SDLK_ESCAPE => running = false,
                        else => {},
                    }
                },
                else => {},
            }
        }
        _ = sdl.SDL_RenderClear(renderer);
        _ = sdl.SDL_RenderCopy(renderer, texture, null, null);
        sdl.SDL_RenderPresent(renderer);
    }
}
