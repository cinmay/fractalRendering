const std = @import("std");
const sdl = @cImport(@cInclude("SDL2/SDL.h"));

const width = 2560;
const height = 1440;

// Declare CUDA functions
extern fn cudaMalloc(ptr: *[*]u16, size: usize) c_int;
extern fn cudaFree(ptr: [*]u16) c_int;
extern fn cudaMemcpy(dst: [*]u16, src: [*]const u16, size: usize, kind: c_int) c_int;
extern fn launch_mandelbrot(pixels: [*]u16) void;

// Save HDR 16-bit image as PPM
pub fn save_to_ppm(pixels: []u16) !void {
    var file = try std.fs.cwd().createFile("mandelbrot_16bit.ppm", .{ .truncate = true });
    defer file.close();

    try file.writer().print("P6\n{} {}\n65535\n", .{ width, height });
    for (pixels) |val| {
        try file.writer().writeAll(&[_]u8{ @truncate(val >> 8), @truncate(val & 0xFF) });
    }
    std.debug.print("✅ Mandelbrot saved to mandelbrot_16bit.ppm\n", .{});
}

// Convert 16-bit color pixels to 8-bit for SDL
fn convert_16bit_to_8bit(src: []u16, dest: []u8) void {
    for (0..(width * height * 3)) |i| {
        dest[i] = @truncate(src[i] >> 8); // Convert 16-bit to 8-bit
    }
}

pub fn main() !void {
    if (sdl.SDL_Init(sdl.SDL_INIT_VIDEO) != 0) {
        std.debug.print("Failed to initialize SDL2: {s}\n", .{sdl.SDL_GetError()});
        return;
    }
    defer sdl.SDL_Quit();

    const window = sdl.SDL_CreateWindow("CUDA Mandelbrot 16-bit", sdl.SDL_WINDOWPOS_CENTERED, sdl.SDL_WINDOWPOS_CENTERED, width, height, sdl.SDL_WINDOW_SHOWN);
    if (window == null) return;
    defer sdl.SDL_DestroyWindow(window);

    const renderer = sdl.SDL_CreateRenderer(window, -1, sdl.SDL_RENDERER_ACCELERATED);
    if (renderer == null) return;
    defer sdl.SDL_DestroyRenderer(renderer);

    const pixels_size = width * height * 3 * 2; // 16-bit per channel
    const pixels_16bit = try std.heap.page_allocator.alloc(u16, pixels_size / 2);
    defer std.heap.page_allocator.free(pixels_16bit);

    const pixels_8bit = try std.heap.page_allocator.alloc(u8, pixels_size / 2);
    defer std.heap.page_allocator.free(pixels_8bit);

    // Allocate CUDA memory
    var cuda_pixels: [*]u16 = undefined;
    if (cudaMalloc(&cuda_pixels, pixels_size) != 0) {
        std.debug.print("❌ CUDA malloc failed! Not enough memory.\n", .{});
        return;
    }
    std.debug.print("✅ CUDA malloc successful: {d} bytes allocated.\n", .{pixels_size});
    defer _ = cudaFree(cuda_pixels);

    // Run Mandelbrot computation on CUDA
    launch_mandelbrot(cuda_pixels);

    // Copy result back from CUDA to Zig
    const cudaMemcpyDeviceToHost = 2;
    if (cudaMemcpy(pixels_16bit.ptr, cuda_pixels, pixels_size, cudaMemcpyDeviceToHost) != 0) {
        std.debug.print("❌ CUDA memcpy failed! Memory copy issue.\n", .{});
        return;
    }
    std.debug.print("✅ CUDA memcpy successful.\n", .{});

    // Convert 16-bit pixels to 8-bit for SDL rendering
    convert_16bit_to_8bit(pixels_16bit, pixels_8bit);

    // Display in SDL2 window
    const texture = sdl.SDL_CreateTexture(renderer, sdl.SDL_PIXELFORMAT_RGB24, sdl.SDL_TEXTUREACCESS_STREAMING, width, height);
    if (texture == null) {
        std.debug.print("❌ Failed to create SDL texture: {s}\n", .{sdl.SDL_GetError()});
        return;
    }
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
