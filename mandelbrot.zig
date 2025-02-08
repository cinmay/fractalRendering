const std = @import("std");
const sdl = @cImport(@cInclude("SDL2/SDL.h"));

const width = 2560; // Screen width
const height = 1440; // Screen height
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

// HDR Color mapping (16-bit precision)
fn hdr_color_mapping(iter: u8) [3]u16 {
    if (iter == max_iter) {
        return [3]u16{ 0, 0, 0 }; // Deep black background
    }

    const t: f64 = @as(f64, @floatFromInt(iter)) / @as(f64, max_iter);

    // Vibrant OLED HDR colors: Orange → Pink → Bright Red
    const r: u16 = @intFromFloat(65535.0 * (1.0 - t * 0.2)); // Deep red
    const g: u16 = @intFromFloat(35000.0 * (1.0 - t * 0.7)); // Orange fading into pink
    const b: u16 = @intFromFloat(40000.0 * t); // Blueish-pinkish hue

    return [3]u16{ r, g, b };
}

// Convert HDR colors (16-bit) to 8-bit for SDL rendering
fn convert_hdr_to_8bit(hdr: [3]u16) [3]u8 {
    return [3]u8{
        @intCast(hdr[0] >> 8), // Convert 16-bit to 8-bit
        @intCast(hdr[1] >> 8),
        @intCast(hdr[2] >> 8),
    };
}

// Save HDR image as PPM (16-bit per channel)
fn save_to_ppm(pixels: []u16) !void {
    var file = try std.fs.cwd().createFile("mandelbrot_hdr.ppm", .{ .truncate = true });
    defer file.close();

    try file.writer().print("P6\n{} {}\n65535\n", .{ width, height });
    for (pixels) |val| {
        try file.writer().writeAll(&[_]u8{ @truncate(val >> 8), @truncate(val & 0xFF) });
    }
}

pub fn main() !void {
    if (sdl.SDL_Init(sdl.SDL_INIT_VIDEO) != 0) {
        std.debug.print("Failed to initialize SDL2: {s}\n", .{sdl.SDL_GetError()});
        return;
    }
    defer sdl.SDL_Quit();

    const window = sdl.SDL_CreateWindow("Mandelbrot HDR 2560x1440", sdl.SDL_WINDOWPOS_CENTERED, sdl.SDL_WINDOWPOS_CENTERED, width, height, sdl.SDL_WINDOW_SHOWN);
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

    // Allocate buffers
    const pixels_size = width * height * 3;
    var pixels_hdr = try std.heap.page_allocator.alloc(u16, pixels_size);
    var pixels_sdl = try std.heap.page_allocator.alloc(u8, pixels_size);
    defer std.heap.page_allocator.free(pixels_hdr);
    defer std.heap.page_allocator.free(pixels_sdl);

    // Compute Mandelbrot set with HDR colors
    for (0..height) |py| {
        for (0..width) |px| {
            const x0 = @as(f64, @floatFromInt(py)) / @as(f64, height) * 3.0 - 2.0; // [-2.0, 1.0] range
            const y0 = @as(f64, @floatFromInt(px)) / @as(f64, width) * 3.0 - 1.5; // [-1.5, 1.5] range

            const iter = mandelbrot(x0, y0);
            const color_hdr = hdr_color_mapping(iter);
            const color_8bit = convert_hdr_to_8bit(color_hdr);

            const index = (py * width + px) * 3;
            pixels_hdr[index] = color_hdr[0];
            pixels_hdr[index + 1] = color_hdr[1];
            pixels_hdr[index + 2] = color_hdr[2];

            pixels_sdl[index] = color_8bit[0];
            pixels_sdl[index + 1] = color_8bit[1];
            pixels_sdl[index + 2] = color_8bit[2];
        }
    }

    // Save HDR image
    try save_to_ppm(pixels_hdr);

    // Display in SDL2 window using 8-bit format (since SDL2 doesn't support HDR)
    const texture = sdl.SDL_CreateTexture(renderer, sdl.SDL_PIXELFORMAT_RGB24, sdl.SDL_TEXTUREACCESS_STATIC, width, height);
    if (texture == null) {
        std.debug.print("Failed to create texture: {s}\n", .{sdl.SDL_GetError()});
        return;
    }
    defer sdl.SDL_DestroyTexture(texture);

    _ = sdl.SDL_UpdateTexture(texture, null, pixels_sdl.ptr, width * 3);

    // Event loop with Q and Esc to exit
    var running = true;
    var event: sdl.SDL_Event = undefined;

    while (running) {
        while (sdl.SDL_PollEvent(&event) != 0) {
            switch (event.type) {
                sdl.SDL_QUIT => running = false,
                sdl.SDL_KEYDOWN => {
                    switch (event.key.keysym.sym) {
                        sdl.SDLK_q, sdl.SDLK_ESCAPE => {
                            running = false; // Exit on Q or Escape
                        },
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
