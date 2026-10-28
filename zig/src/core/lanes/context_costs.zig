//! Optional measured allocation costs for explicit context ranges and concurrent stream counts.
const std = @import("std");
const Table = @import("table.zig").Table;
const alloc = @import("allocate.zig");

/// Context lengths include all proposed rows; both range endpoints are inclusive, and row one is forward_ms[0].
pub const Profile = struct {
    min_context: u64 = 0,
    max_context: u64,
    streams: u32,
    forward_ms: []const f64,
    overhead_ms: ?f64 = null,
};

pub const Band = struct {
    min_context: u64,
    max_context: u64,
    streams: u32,
    forward: Table,
    overhead_ms: ?f64,
};

pub const Choice = struct { forward: *const Table, overhead_ms: f64 };

pub const Catalog = struct {
    bands: []Band = &.{},

    pub fn init(gpa: std.mem.Allocator, profiles: []const Profile, rows: u32, streams: u32) !Catalog {
        const out: Catalog = .{ .bands = try gpa.alloc(Band, profiles.len) };
        var ready: usize = 0;
        errdefer {
            for (out.bands[0..ready]) |*band| band.forward.deinit(gpa);
            gpa.free(out.bands);
        }
        for (profiles) |profile| {
            if (profile.max_context == 0 or profile.min_context > profile.max_context or
                profile.streams == 0 or profile.streams > streams or
                profile.forward_ms.len < profile.streams or profile.forward_ms.len > rows)
                return error.BadContextCosts;
            for (profile.forward_ms) |ms| if (!std.math.isFinite(ms) or ms <= 0) return error.BadContextCosts;
            if (profile.overhead_ms) |ms| if (!std.math.isFinite(ms) or ms < 0) return error.BadContextCosts;
            const slots = try gpa.alloc(?f64, profile.forward_ms.len + 1);
            slots[0] = null;
            for (profile.forward_ms, 1..) |ms, row| slots[row] = ms;
            out.bands[ready] = .{
                .min_context = profile.min_context,
                .max_context = profile.max_context,
                .streams = profile.streams,
                .forward = .{ .slots = slots },
                .overhead_ms = profile.overhead_ms,
            };
            ready += 1;
        }
        std.mem.sort(Band, out.bands, {}, struct {
            fn less(_: void, a: Band, b: Band) bool {
                if (a.streams != b.streams) return a.streams < b.streams;
                return a.max_context < b.max_context;
            }
        }.less);
        for (out.bands, 0..) |band, i| {
            if (i == 0) continue;
            const previous = out.bands[i - 1];
            if (previous.streams == band.streams and band.min_context <= previous.max_context)
                return error.OverlappingContextCosts;
        }
        return out;
    }

    pub fn deinit(c: *Catalog, gpa: std.mem.Allocator) void {
        for (c.bands) |*band| band.forward.deinit(gpa);
        gpa.free(c.bands);
        c.* = .{};
    }

    /// A measured range applies only when it covers every feasible width; no interpolation or extrapolation.
    pub fn select(c: *const Catalog, context: u64, streams: usize, rows: u64) ?*const Band {
        for (c.bands) |*band| {
            if (band.streams != streams or context < band.min_context or context > band.max_context) continue;
            if (rows == 0 or rows >= band.forward.slots.len) return null;
            return band;
        }
        return null;
    }

    pub fn resolve(c: *const Catalog, context: u64, streams: usize, rows: u64, fallback: *const Table, overhead: f64) Choice {
        return c.resolveSpan(context, context, streams, rows, fallback, overhead);
    }

    /// The pending-only and widest proposed contexts must stay in one calibrated band.
    pub fn resolveSpan(c: *const Catalog, start: u64, end: u64, streams: usize, rows: u64, fallback: *const Table, overhead: f64) Choice {
        if (start <= end) {
            if (c.select(end, streams, rows)) |band| {
                if (start >= band.min_context)
                    return .{ .forward = &band.forward, .overhead_ms = band.overhead_ms orelse overhead };
            }
        }
        return .{ .forward = fallback, .overhead_ms = overhead };
    }
};

test "context costs select explicit ranges and preserve fallback outside coverage" {
    const gpa = std.testing.allocator;
    var catalog = try Catalog.init(gpa, &.{
        .{ .min_context = 65536, .max_context = 131072, .streams = 1, .forward_ms = &.{ 30, 31, 32, 33, 34, 35 } },
        .{ .min_context = 2048, .max_context = 4096, .streams = 1, .forward_ms = &.{ 10, 11, 12, 13, 14, 15 }, .overhead_ms = 2 },
        .{ .min_context = 2048, .max_context = 4096, .streams = 2, .forward_ms = &.{ 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32 } },
    }, 16, 2);
    defer catalog.deinit(gpa);
    const fallback: Table = .{};
    const first = catalog.resolve(2048, 1, 6, &fallback, 8);
    try std.testing.expectEqual(@as(?f64, 15), first.forward.get(6));
    try std.testing.expectEqual(@as(f64, 2), first.overhead_ms);
    const last = catalog.resolve(131072, 1, 6, &fallback, 8);
    try std.testing.expectEqual(@as(?f64, 35), last.forward.get(6));
    try std.testing.expectEqual(@as(f64, 8), last.overhead_ms);
    const shared = catalog.resolve(4096, 2, 12, &fallback, 8);
    try std.testing.expectEqual(@as(?f64, 32), shared.forward.get(12));
    for ([_]u64{ 2047, 4097, 65535, 131073 }) |context| {
        const choice = catalog.resolve(context, 1, 6, &fallback, 8);
        try std.testing.expect(choice.forward == &fallback);
        try std.testing.expectEqual(@as(f64, 8), choice.overhead_ms);
    }
    try std.testing.expect(catalog.resolve(2048, 1, 7, &fallback, 8).forward == &fallback);
    try std.testing.expect(catalog.resolve(2048, 2, 13, &fallback, 8).forward == &fallback);
    try std.testing.expect(catalog.resolve(2048, 3, 6, &fallback, 8).forward == &fallback);
    try std.testing.expect(catalog.resolveSpan(2046, 2051, 1, 6, &fallback, 8).forward == &fallback);
    try std.testing.expect(catalog.resolveSpan(4095, 4100, 1, 6, &fallback, 8).forward == &fallback);
    try std.testing.expect(catalog.resolveSpan(2051, 2048, 1, 6, &fallback, 8).forward == &fallback);
    try std.testing.expect(catalog.resolveSpan(2048, 2053, 1, 6, &fallback, 8).forward != &fallback);
}

test "calibrated context costs change the chosen depth while the empty catalog preserves it" {
    const gpa = std.testing.allocator;
    var fallback = try alloc.extendCosts(gpa, &.{ .{ .width = 1, .ms = 10 }, .{ .width = 3, .ms = 30 } }, 3);
    defer fallback.deinit(gpa);
    var catalog = try Catalog.init(gpa, &.{
        .{ .min_context = 2048, .max_context = 4096, .streams = 1, .forward_ms = &.{ 10, 11, 12 }, .overhead_ms = 0 },
        .{ .min_context = 65536, .max_context = 131072, .streams = 1, .forward_ms = &.{ 10, 25, 40 }, .overhead_ms = 0 },
    }, 3, 1);
    defer catalog.deinit(gpa);
    const empty: Catalog = .{};
    for ([_]struct { catalog: *const Catalog, context: u64, count: u32 }{
        .{ .catalog = &empty, .context = 2048, .count = 0 },
        .{ .catalog = &catalog, .context = 2048, .count = 2 },
        .{ .catalog = &catalog, .context = 65536, .count = 0 },
        .{ .catalog = &catalog, .context = 8192, .count = 0 },
    }) |case| {
        const choice = case.catalog.resolve(case.context, 1, 3, &fallback, 0);
        const counts = try alloc.allocate(gpa, &.{1}, &.{&.{ 0.9, 0.8 }}, choice.forward, choice.overhead_ms, 3);
        defer gpa.free(counts);
        try std.testing.expectEqual(case.count, counts[0]);
    }
}

test "context costs own their values and reject ambiguous or invalid calibration" {
    const gpa = std.testing.allocator;
    var times = [_]f64{ 10, 11, 12 };
    const profile: Profile = .{ .min_context = 1, .max_context = 4096, .streams = 1, .forward_ms = &times };
    var catalog = try Catalog.init(gpa, &.{profile}, 3, 1);
    defer catalog.deinit(gpa);
    times[0] = 99;
    try std.testing.expectEqual(@as(?f64, 10), catalog.select(1, 1, 3).?.forward.get(1));
    try std.testing.expectError(error.OverlappingContextCosts, Catalog.init(gpa, &.{ profile, profile }, 3, 1));
    for ([_]f64{ 0, -1, std.math.inf(f64), std.math.nan(f64) }) |invalid| {
        times[0] = invalid;
        try std.testing.expectError(error.BadContextCosts, Catalog.init(gpa, &.{profile}, 3, 1));
    }
    times[0] = 10;
    var bad = profile;
    bad.overhead_ms = -1;
    try std.testing.expectError(error.BadContextCosts, Catalog.init(gpa, &.{bad}, 3, 1));
    bad = profile;
    bad.min_context = 4097;
    try std.testing.expectError(error.BadContextCosts, Catalog.init(gpa, &.{bad}, 3, 1));
    bad = profile;
    bad.streams = 2;
    try std.testing.expectError(error.BadContextCosts, Catalog.init(gpa, &.{bad}, 3, 1));
    bad = profile;
    bad.forward_ms = &.{};
    try std.testing.expectError(error.BadContextCosts, Catalog.init(gpa, &.{bad}, 3, 1));
}
