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
pub fn main() !void {
    if (sdl.SDL_Init(sdl.SDL_INIT_VIDEO) != 0) {
        std.debug.print("Failed to initialize SDL2: {s}\n", .{sdl.SDL_GetError()});
        return;
    }
    defer sdl.SDL_Quit();

    const window = sdl.SDL_CreateWindow("CUDA Mandelbrot 16-bit", sdl.SDL_WINDOWPOS_CENTERED, sdl.SDL_WINDOWPOS_CENTERED, width, height, sdl.SDL_WINDOW_SHOWN);
    if (window == null) return;
    defer sdl.SDL_DestroyWindow(window);

    const pixels_size = width * height * 3 * 2; // 16-bit per channel (R, G, B)
    const pixels = try std.heap.page_allocator.alloc(u16, pixels_size / 2);
    defer std.heap.page_allocator.free(pixels);

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
    if (cudaMemcpy(pixels.ptr, cuda_pixels, pixels_size, cudaMemcpyDeviceToHost) != 0) {
        std.debug.print("❌ CUDA memcpy failed! Memory copy issue.\n", .{});
        return;
    }
    std.debug.print("✅ CUDA memcpy successful.\n", .{});

    // Save output image
    try save_to_ppm(pixels);
}
