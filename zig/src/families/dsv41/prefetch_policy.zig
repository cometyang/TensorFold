//! Optional decode prefetch stages and byte budgets; parsed once before graph capture.
const std = @import("std");

pub const Stage = enum(u5) { attention, moe, next, engram, head };
pub const all_stages: u8 = 31;

pub const Policy = struct {
    mask: u8 = all_stages,
    percent: u8 = 100,

    pub fn parse(mask: ?[]const u8, percent: ?[]const u8) !Policy {
        const m = if (mask) |v| std.fmt.parseInt(u8, v, 0) catch return error.BadPrefetchMask else all_stages;
        const p = if (percent) |v| std.fmt.parseInt(u8, v, 10) catch return error.BadPrefetchPercent else 100;
        if (m & ~all_stages != 0) return error.BadPrefetchMask;
        if (p > 100) return error.BadPrefetchPercent;
        return .{ .mask = m, .percent = p };
    }

    pub fn budget(p: Policy, stage: Stage, original: u64) u64 {
        if (p.mask & (@as(u8, 1) << @as(u3, @intCast(@intFromEnum(stage)))) == 0) return 0;
        return (original * p.percent / 100) & ~@as(u64, 127);
    }
};

pub fn environment() !Policy {
    const mask = if (std.c.getenv("TF_DS_L2_PREFETCH_MASK")) |v| std.mem.span(v) else null;
    const percent = if (std.c.getenv("TF_DS_L2_PREFETCH_BUDGET_PERCENT")) |v| std.mem.span(v) else null;
    return Policy.parse(mask, percent);
}

test "default prefetch policy preserves every served stage budget" {
    const p = try Policy.parse(null, null);
    const stages = [_]Stage{ .attention, .moe, .next, .engram, .head };
    const original = [_]u64{ 16 << 20, 8 << 20, 16 << 20, 4 << 20, 16 << 20 };
    for (stages, original) |stage, bytes| try std.testing.expectEqual(bytes, p.budget(stage, bytes));
}

test "prefetch mask selects destinations and budgets stay on cache lines" {
    const p = try Policy.parse("0x16", "25");
    try std.testing.expectEqual(@as(u64, 0), p.budget(.attention, 16 << 20));
    try std.testing.expectEqual(@as(u64, 2 << 20), p.budget(.moe, 8 << 20));
    try std.testing.expectEqual(@as(u64, 4 << 20), p.budget(.next, 16 << 20));
    try std.testing.expectEqual(@as(u64, 0), p.budget(.engram, 4 << 20));
    try std.testing.expectEqual(@as(u64, 4 << 20), p.budget(.head, 16 << 20));
    try std.testing.expectEqual(@as(u64, 167680), (try Policy.parse(null, "1")).budget(.head, 16 << 20));
    try std.testing.expectEqual(@as(u64, 0), (try Policy.parse("0", null)).budget(.head, 16 << 20));
    try std.testing.expectEqual(@as(u64, 0), (try Policy.parse(null, "0")).budget(.head, 16 << 20));
}

test "invalid prefetch settings fail before graph capture" {
    for ([_][]const u8{ "", "-1", "32", "256", "attention" }) |v|
        try std.testing.expectError(error.BadPrefetchMask, Policy.parse(v, null));
    for ([_][]const u8{ "", "-1", "101", "256", "0.5" }) |v|
        try std.testing.expectError(error.BadPrefetchPercent, Policy.parse(null, v));
}
