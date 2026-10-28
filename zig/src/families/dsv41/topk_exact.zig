//! Opt-in exact radix selection with bounded scratch and deterministic score/index ordering.
const std = @import("std");
const cuda = @import("cuda");

const scratch_bytes = 32 << 20;
pub const Plan = struct { parts: usize, scratch: usize };

pub fn plan(rows: usize, n: usize, k: usize) ?Plan {
    if (rows == 0 or k == 0 or k > 2048 or k > n or n > 1048576) return null;
    if (n <= 16384) return .{ .parts = 1, .scratch = 0 };
    const width: usize = if (rows <= 8) 4096 else 16384;
    const parts = @max((n + 32767) / 32768, @min(16, (n + width - 1) / width));
    if (parts * k > 32768 or rows > scratch_bytes / (parts * k * 8)) return null;
    return .{ .parts = parts, .scratch = rows * parts * k * 8 };
}

pub const Exact = struct {
    keys: cuda.Function,
    indices: cuda.Function,
    key_parts: cuda.Function,
    score_parts: cuda.Function,
    merge: cuda.Function,
    scratch: cuda.DeviceBuffer,

    pub fn load(d: *const cuda.Driver, module: cuda.Module) !?Exact {
        const setting = std.c.getenv("TF_DS_TOPK_EXACT") orelse return null;
        if (!std.mem.eql(u8, std.mem.span(setting), "1")) return null;
        const keys = try module.function("tf_ds_topk_i64_radix_kernel");
        const indices = try module.function("tf_ds_topk_indices_radix_kernel");
        const key_parts = try module.function("tf_ds_topk_i64_parts_kernel");
        const score_parts = try module.function("tf_ds_topk_indices_parts_kernel");
        const merge = try module.function("tf_ds_topk_indices_merge_kernel");
        return .{ .keys = keys, .indices = indices, .key_parts = key_parts, .score_parts = score_parts, .merge = merge, .scratch = try cuda.DeviceBuffer.alloc(d, scratch_bytes) };
    }

    pub fn run(t: *const Exact, s: cuda.Stream, input: u64, stride: usize, rows: usize, n: usize, k: usize, vis: ?u64, out: u64) !bool {
        const shape = plan(rows, n, k) orelse return false;
        var a: cuda.Args = .{};
        a.add(input);
        a.add(@as(c_longlong, @intCast(stride)));
        a.add(@as(c_int, @intCast(n)));
        a.add(@as(c_int, @intCast(k)));
        var function = if (vis != null) t.indices else t.keys;
        if (shape.parts > 1) {
            a.add(@as(c_int, @intCast(shape.parts)));
            a.add(t.scratch.ptr);
            function = if (vis != null) t.score_parts else t.key_parts;
        } else {
            if (vis) |value| a.add(value);
            a.add(out);
        }
        try cuda.launch.launch(function, .{
            .grid = .{ .x = @intCast(rows), .y = @intCast(shape.parts) },
            .block = .{ .x = 1024 },
        }, s, &a);
        if (shape.parts > 1) {
            a = .{};
            a.add(t.scratch.ptr);
            a.add(@as(c_longlong, @intCast(shape.parts * k)));
            a.add(@as(c_int, @intCast(shape.parts * k)));
            a.add(@as(c_int, @intCast(k)));
            if (vis) |value| a.add(value);
            a.add(out);
            try cuda.launch.launch(if (vis != null) t.merge else t.keys, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, s, &a);
        }
        return true;
    }
};

test "selection tiles and merges remain inside their register and scratch bounds" {
    for ([_]usize{ 1, 6, 16, 128, 2048 }) |rows| {
        for ([_]usize{ 512, 513, 16384, 16385, 32768, 65536, 131072, 1048576 }) |n| {
            for ([_]usize{ 1, 511, 512, 2048 }) |k| {
                if (plan(rows, n, k)) |p| {
                    try std.testing.expect((n + p.parts - 1) / p.parts <= 32768);
                    try std.testing.expect(p.parts * k <= 32768);
                    try std.testing.expect(p.scratch <= scratch_bytes);
                    try std.testing.expect(k <= n);
                }
            }
        }
    }
    try std.testing.expectEqual(@as(usize, 8), plan(1, 32768, 512).?.parts);
    try std.testing.expectEqual(@as(usize, 2), plan(128, 32768, 512).?.parts);
    try std.testing.expect(plan(1, 1, 0) == null);
    try std.testing.expect(plan(1, 513, 2048) == null);
    try std.testing.expect(plan(1, 1048577, 512) == null);
}
