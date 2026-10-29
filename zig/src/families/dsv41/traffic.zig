//! Optional aggregate hardware counters around a real verification round, with no kernel replay.
const std = @import("std");

const Begin = *const fn (u64, u32, c_int) callconv(.c) c_int;
const End = *const fn () callconv(.c) c_int;
const Finish = *const fn () callconv(.c) void;

pub const Collector = struct {
    lib: ?std.DynLib = null,
    begin_fn: ?Begin = null,
    end_fn: ?End = null,
    finish_fn: ?Finish = null,

    pub fn open() Collector {
        const path = std.c.getenv("TF_DS_TRAFFIC_LIBRARY") orelse return .{};
        var lib = std.DynLib.open(std.mem.span(path)) catch {
            std.log.err("traffic collector library unavailable", .{});
            return .{};
        };
        const begin_fn = lib.lookup(Begin, "tf_traffic_begin");
        const end_fn = lib.lookup(End, "tf_traffic_end");
        const finish_fn = lib.lookup(Finish, "tf_traffic_finish");
        if (begin_fn == null or end_fn == null or finish_fn == null) {
            std.log.err("traffic collector ABI unavailable", .{});
            lib.close();
            return .{};
        }
        return .{ .lib = lib, .begin_fn = begin_fn, .end_fn = end_fn, .finish_fn = finish_fn };
    }

    pub fn begin(c: *Collector, positions: []const i64, rows: usize, rank: u32) bool {
        const f = c.begin_fn orelse return false;
        var context: u64 = 0;
        for (positions) |p| context = @max(context, @as(u64, @intCast(@max(p, 0))));
        return f(context, @intCast(rows), @intCast(rank)) == 1;
    }

    pub fn end(c: *Collector) void {
        if (c.end_fn) |f| _ = f();
    }

    pub fn finish(c: *Collector) void {
        if (c.finish_fn) |f| f();
    }

    pub fn close(c: *Collector) void {
        c.finish();
        if (c.lib) |*lib| lib.close();
        c.* = .{};
    }
};
