const std = @import("std");
const sdl = @cImport(@cInclude("SDL2/SDL.h"));

const width = 800;
const height = 600;
const max_iter = 255;

// Mandelbrot function
fn mandelbrot(x: f64, y: f64) u8 {
    var zx: f64 = 0;
    var zy: f64 = 0;
    var iter: u8 = 0;

    while (zx * zx + zy * zy < 4 and iter < max_iter) : (iter += 1) {
        const temp = zx * zx - zy * zy + x;
        zy = 2 * zx * zy + y;
        zx = temp;
    }
    return iter;
}

// Color mapping function
fn color_mapping(iter: u8) [3]u8 {
    if (iter == max_iter) {
        return [3]u8{ 0, 0, 0 }; // Deep black background
    }
    const t: f64 = @as(f64, @floatFromInt(iter)) / @as(f64, max_iter);
    const r: u8 = @intFromFloat(255 * (1.0 - t * 0.3)); // Red dominates
    const g: u8 = @intFromFloat(120 * (1.0 - t * 0.7)); // Dark Orange fades to pink
    const b: u8 = @intFromFloat(200 * (t)); // Blueish tint for pink transition
    return [3]u8{ r, g, b };
}

// Save PPM image
fn save_to_ppm(pixels: []u8) !void {
    var file = try std.fs.cwd().createFile("mandelbrot.ppm", .{ .truncate = true });
    defer file.close();
    try file.writer().print("P6\n{} {}\n255\n", .{ width, height });
    try file.writer().writeAll(pixels);
}

pub fn main() !void {
    if (sdl.SDL_Init(sdl.SDL_INIT_VIDEO) != 0) {
        std.debug.print("Failed to initialize SDL2: {s}\n", .{sdl.SDL_GetError()});
        return;
    }
    defer sdl.SDL_Quit();

    const window = sdl.SDL_CreateWindow("Mandelbrot Rotated 90°", sdl.SDL_WINDOWPOS_CENTERED, sdl.SDL_WINDOWPOS_CENTERED, width, height, sdl.SDL_WINDOW_SHOWN);
    if (window == null) {
        std.debug.print("Failed to create window: {s}\n", .{sdl.SDL_GetError()});
        return;
    }
    defer sdl.SDL_DestroyWindow(window);

    const renderer = sdl.SDL_CreateRenderer(window, -1, sdl.SDL_RENDERER_ACCELERATED);
    if (renderer == null) {
        std.debug.print("Failed to create renderer: {s}\n", .{sdl.SDL_GetError()});
        return;
    }
    defer sdl.SDL_DestroyRenderer(renderer);

    const pixels_size = width * height * 3;
    var pixels = try std.heap.page_allocator.alloc(u8, pixels_size);
    defer std.heap.page_allocator.free(pixels);

    // Compute Mandelbrot set with 90-degree counterclockwise rotation
    for (0..height) |py| {
        for (0..width) |px| {
            const x0 = @as(f64, @floatFromInt(py)) / @as(f64, height) * 3.5 - 2.5;
            const y0 = @as(f64, @floatFromInt(px)) / @as(f64, width) * 2.0 - 1.0;
            const iter = mandelbrot(x0, y0);
            const color = color_mapping(iter);

            const index = (py * width + px) * 3;
            pixels[index] = color[0];
            pixels[index + 1] = color[1];
            pixels[index + 2] = color[2];
        }
    }

    // Save the image
    try save_to_ppm(pixels);

    // Display in SDL2 window
    const texture = sdl.SDL_CreateTexture(renderer, sdl.SDL_PIXELFORMAT_RGB24, sdl.SDL_TEXTUREACCESS_STATIC, width, height);
    if (texture == null) {
        std.debug.print("Failed to create texture: {s}\n", .{sdl.SDL_GetError()});
        return;
    }
    defer sdl.SDL_DestroyTexture(texture);

    _ = sdl.SDL_UpdateTexture(texture, null, pixels.ptr, width * 3);

    // Event loop with key handling for exiting
    var running = true;
    var event: sdl.SDL_Event = undefined;

    while (running) {
        while (sdl.SDL_PollEvent(&event) != 0) {
            switch (event.type) {
                sdl.SDL_QUIT => running = false,
                sdl.SDL_KEYDOWN => {
                    switch (event.key.keysym.sym) {
                        sdl.SDLK_q, sdl.SDLK_ESCAPE => running = false, // Exit on Q or Escape
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
