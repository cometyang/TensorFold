//! A round's windows (Python `_plan_window`, `_copy_proposal`, `_allocate`): forced, copied or drafted rows.
const std = @import("std");
const accept = @import("accept.zig");
const alloc = @import("allocate.zig");
const be = @import("backend.zig");
const depth = @import("depth.zig");
const fill = @import("fill.zig");
const shape = @import("shape.zig");
const ev = @import("events.zig");
const trail = @import("trail.zig");
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const Value = ev.Value;
const f = ev.f;

pub const Kind = enum { forced, copy, head, fill, none };

/// One stream's window as planned, then as allocated (Python's plan lists).
pub const Plan = struct {
    stream: *Stream,
    position: u64, // the stream's cache length when the round started
    kind: Kind,
    held: u32 = 0, // head drafts the backend holds
    tokens: []const u32 = &.{}, // drafts known on the host (forced, copied, tree)
    forced: []const u32 = &.{},
    parents: ?[]const i32 = null, // a tree's drafts' parents
    branch_from: u32 = 0, // the first row of suffix-match branches (0: none; lanes/fill.zig)
    tried: bool = false, // the planner read the head's chain back to graft branches
    lanes: ?*const shape.Shape = null, // the held drafts' tree shape (each lane's depth and rank)

    pub fn count(p: Plan) usize {
        return p.held + p.tokens.len;
    }
};

/// Forced tokens first, then a copied continuation, then the head's drafts (Python `_plan_window`).
pub fn plan(e: *Engine, s: *Stream, copied: ?[]const u32) !Plan {
    const a = e.arena.allocator();
    var p: Plan = .{ .stream = s, .position = s.cache_len, .kind = .none };
    const queued = s.next;
    s.next = null;
    defer if (queued) |q| {
        if (q.tokens) |t| e.gpa.free(t);
        if (q.parents) |parents| e.gpa.free(parents);
    };
    if (s.force.items.len > 0) {
        const width: usize = if (s.drafts) @min(e.cfg.base_width, e.cfg.batch_rows) else 1;
        const take = @min(width - 1, s.force.items.len);
        const forced = try a.dupe(u32, s.force.items[0..take]);
        s.force.replaceRangeAssumeCapacity(0, take, &.{});
        p.kind = .forced;
        p.forced = forced;
        p.tokens = forced;
    } else {
        var copy: []const u32 = &.{};
        if (copied) |c| copy = c else if (s.drafts) copy = try copyProposal(e, s);
        if (copy.len > 0) {
            p.kind = .copy;
            p.tokens = try a.dupe(u32, copy);
        } else if (s.drafts and queued != null) {
            const q = queued.?;
            if (q.tokens) |t| {
                p.tokens = try a.dupe(u32, t);
                if (q.parents) |parents| {
                    if (!accept.isChain(parents)) p.parents = try a.dupe(i32, parents);
                }
            } else {
                p.held = q.count;
                if (s.lanes) |*l| {
                    // a held tree is a lone stream's (shared rounds verify chains); a chain's lanes are observed too
                    if (q.parents) |parents| {
                        if (e.alone) p.parents = try a.dupe(i32, parents) else p.held = 0;
                    }
                    if (p.held > 0) p.lanes = l;
                }
            }
            if (p.count() > 0) p.kind = .head;
        }
    }
    if (e.cfg.fill_lanes > 0 and s.drafts and p.kind != .forced and (p.parents == null or p.lanes != null)) try fillLanes(e, s, &p);
    try trail.event(e, &.{ f("ev", trail.str("plan")), f("stream", trail.str(s.id)), f("kind", trail.str(@tagName(p.kind))), f("n", trail.int(p.count())), f("tree", .{ .bool = p.parents != null }) });
    return p;
}

/// Suffix-match branches grafted onto the planned drafts, up to cfg.fill_lanes rows.
fn fillLanes(e: *Engine, s: *Stream, p: *Plan) !void {
    const a = e.arena.allocator();
    const first = p.count();
    const room = @min(@as(usize, e.cfg.fill_lanes) -| (1 + first), s.graft_room orelse std.math.maxInt(u32));
    if (room == 0) return;
    const found = try fill.suffixCandidates(a, s.context.items, 16, e.cfg.fill_match, room, 8);
    if (found.len == 0 or (s.graft_hit < e.cfg.fill_floor and s.rounds % 8 != 0)) return;
    // the trunk's lanes: the drafts in order for a chain, a tree's rank-0 path from the pending row
    var trunk_lanes: [shape.max_depth * shape.ranks * 4]usize = undefined;
    const levels = if (p.lanes) |l| shape.chainLanes(l.*, &trunk_lanes) else first;
    if (p.lanes == null) {
        for (trunk_lanes[0..first], 0..) |*x, k| x.* = k;
    }
    const trunk = try headTrunk(e, s, p.*, levels) orelse return;
    p.tried = p.held > 0;
    const b = try fill.graft(a, trunk, found, room);
    if (b.tokens.len == 0) return;
    const parents = try a.alloc(i32, first + b.tokens.len);
    if (p.parents) |held| {
        @memcpy(parents[0..first], held);
    } else {
        for (parents[0..first], 0..) |*q, k| q.* = @as(i32, @intCast(k)) - 1;
    }
    // a graft's parent: the pending row, a trunk level (that level's lane), or one of its own rows (after the drafts)
    for (parents[first..], b.parents) |*q, g| q.* = if (g < 0) -1 else if (g < levels) @intCast(trunk_lanes[@intCast(g)]) else @intCast(first + @as(usize, @intCast(g)) - levels);
    p.tokens = if (p.held > 0) b.tokens else try std.mem.concat(a, u32, &.{ p.tokens, b.tokens });
    p.parents = parents;
    p.branch_from = @intCast(1 + first);
    if (p.kind == .none) p.kind = .fill;
}

/// The planned drafts' trunk of `levels` lanes: host drafts as they are, held ones with the head's ranks (null: unread).
fn headTrunk(e: *Engine, s: *Stream, p: Plan, levels: usize) !?fill.Trunk {
    if (p.held == 0) return .{ .tokens = p.tokens };
    const a = e.arena.allocator();
    const read = e.backend.vtable.alternatives orelse return null;
    const alts = try a.alloc(be.Alternative, levels);
    if (try read(e.backend.ptr, s, alts) < levels) return null;
    const chain = try a.alloc(u32, levels);
    const others = try a.alloc([3]u32, levels);
    for (alts, chain, others) |x, *t, *o| {
        t.* = x.tokens[0];
        o.* = x.tokens[1..4].*;
    }
    return .{ .tokens = chain, .others = others };
}

/// Copied spans backed by `enter_match` matching tokens (Python `_copy_proposal`); the proposer owns them.
pub fn copyProposal(e: *Engine, s: *Stream) ![]const u32 {
    const p = s.proposer orelse return &.{};
    if (e.cfg.max_copy == 0 or s.force.items.len > 0) return &.{};
    const width: u32 = if (e.alone) (s.copy_width orelse e.cfg.first_copy) else e.cfg.first_copy;
    const copied = p.propose(s.context.items, @min(@as(i64, width), s.draftRoom() - 1)) catch return &.{};
    if (copied.len < 2 or p.lastMatch() < e.cfg.enter_match) return &.{};
    return copied;
}

pub fn copyAhead(ptr: *anyopaque, w: *const depth.Who) bool {
    const e: *Engine = @ptrCast(@alignCast(ptr));
    const s: *Stream = @ptrCast(@alignCast(w.stream.?));
    const copied = copyProposal(e, s) catch return false;
    return copied.len > 0;
}

pub fn probe(e: *Engine) depth.Probe {
    return .{ .ptr = e, .copy = copyAhead };
}

pub fn who(s: *Stream) depth.Who {
    return .{ .state = &s.depth, .draft_room = s.draftRoom(), .finished = s.finished, .forced = s.force.items.len > 0, .granted = s.granted, .stream = s };
}

/// Trim draft prefixes by landing probability to the shared width with the most expected tokens a ms.
pub fn allocate(e: *Engine, plans: []Plan) !void {
    const a = e.arena.allocator();
    const fixed = try a.alloc(u32, plans.len);
    const probs = try a.alloc([]const f64, plans.len);
    var context_start: u64 = 0;
    var context_end: u64 = 0;
    for (plans, fixed, probs) |p, *fx, *pr| {
        const n = p.count();
        context_start = @max(context_start, p.position +| 1);
        context_end = @max(context_end, p.position +| 1 +| @as(u64, @intCast(n)));
        if (p.kind == .forced or n == 0) {
            fx.* = 1 + @as(u32, @intCast(if (p.kind == .forced) n else 0));
            pr.* = &.{};
        } else if (p.kind == .copy) {
            fx.* = 1;
            pr.* = try alloc.chainProbabilities(a, &.{e.cfg.copy_rate}, n);
        } else {
            fx.* = 1;
            pr.* = try nodeChances(e, p.stream, n);
        }
    }
    const max_rows = @max(plans.len, e.cfg.batch_rows);
    var possible_rows: u64 = 0;
    for (fixed, probs) |n, p| possible_rows += n + @as(u64, @intCast(p.len));
    const choice = e.cfg.context_costs.resolveSpan(
        context_start,
        context_end,
        plans.len,
        @min(possible_rows, max_rows),
        &e.cfg.shared_costs,
        e.rule.overheadMs(plans.len),
    );
    const counts = try alloc.allocate(a, fixed, probs, choice.forward, choice.overhead_ms, max_rows);
    for (plans, probs, counts) |*p, chances, c| {
        if (p.kind == .head) p.stream.granted = c;
        if (chances.len == 0 or c == chances.len) continue;
        if (c == 0) {
            p.kind = .none;
            p.held = 0;
            p.tokens = &.{};
            p.parents = null;
        } else if (p.parents) |parents| {
            const tree = try accept.sanitizeTree(a, p.tokens, parents, c);
            p.tokens = tree.tokens;
            p.parents = tree.parents;
        } else if (p.held > 0) {
            p.held = c;
        } else p.tokens = p.tokens[0..c];
    }
    const rows = try a.alloc(Value, plans.len);
    for (rows, plans) |*r, p| {
        const pair = try a.alloc(Value, 2);
        pair[0] = trail.str(@tagName(p.kind));
        pair[1] = trail.int(p.count());
        r.* = .{ .list = pair };
    }
    try trail.event(e, &.{ f("ev", trail.str("alloc")), f("plans", .{ .list = rows }) });
}

/// The head's chance for each held draft, else chain chances from the stream's per-depth acceptance.
pub fn nodeChances(e: *Engine, s: *Stream, n: usize) ![]const f64 {
    const a = e.arena.allocator();
    if (e.cfg.node_probabilities) {
        if (e.backend.vtable.probabilities) |chances| {
            const out = try a.alloc(f64, n);
            if (try chances(e.backend.ptr, s, out)) return out;
        }
    }
    return alloc.chainProbabilities(a, try e.rule.rates(who(s)), n);
}

/// The window's rows, parents and keyed positions, ready for the backend.
pub fn build(e: *Engine, p: Plan, early: bool) !be.Window {
    const a = e.arena.allocator();
    const rows = 1 + p.count();
    const parents = try accept.rowParents(a, rows, p.parents);
    const ds = try accept.depths(a, parents);
    const positions = try a.alloc(u64, rows);
    for (positions, ds) |*x, d| x.* = p.position + 1 + d;
    return .{ .stream = p.stream, .pending = p.stream.pending.?, .held = p.held, .tokens = p.tokens, .parents = if (p.parents != null) parents else null, .positions = positions, .early = early };
}

pub fn rowParents(e: *Engine, w: be.Window) ![]const i32 {
    if (w.parents) |p| return p;
    return accept.rowParents(e.arena.allocator(), w.rows(), null);
}

/// The rows' tokens: the pending one, then the drafts (held drafts as the backend read them back).
pub fn tokens(e: *Engine, w: be.Window, o: be.Verified) ![]u32 {
    const out = try e.arena.allocator().alloc(u32, w.rows());
    out[0] = w.pending;
    @memcpy(out[1 .. 1 + w.held], o.drafts[0..w.held]);
    @memcpy(out[1 + w.held ..], w.tokens);
    return out;
}

test "allocation selects the longest proposed end and the actual stream count" {
    const Config = @import("config.zig").Config;
    const fake = @import("fake.zig");
    const gpa = std.testing.allocator;
    var cfg = try Config.init(gpa, .{
        .exact_width = 6,
        .drafts = 5,
        .draft_prior = &.{ 0.9, 0.8 },
        .hidden_rows = true,
        .max_streams = 2,
        .batch_rows = 16,
        .window_costs = &.{ .{ .width = 1, .ms = 10 }, .{ .width = 6, .ms = 60 } },
        .context_costs = &.{
            .{ .min_context = 2048, .max_context = 4096, .streams = 1, .forward_ms = &.{ 10, 11, 12, 13, 14, 15 }, .overhead_ms = 0 },
            .{ .min_context = 2048, .max_context = 4096, .streams = 2, .forward_ms = &.{ 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 }, .overhead_ms = 0 },
            .{ .min_context = 65536, .max_context = 131072, .streams = 2, .forward_ms = &.{ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120 }, .overhead_ms = 0 },
        },
    }, 6, 5);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var e = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    try e.rule.overhead.append(gpa, .{ .streams = 1, .ms = 0 });
    try e.rule.overhead.append(gpa, .{ .streams = 2, .ms = 0 });
    var first = try Stream.init(gpa, .{ .id = "first", .prompt = &.{1}, .max_new = 32, .eos = &.{} });
    defer first.deinit(gpa);
    var second = try Stream.init(gpa, .{ .id = "second", .prompt = &.{1}, .max_new = 32, .eos = &.{} });
    defer second.deinit(gpa);
    for ([_]struct { positions: []const u64, expected: usize }{
        .{ .positions = &.{2045}, .expected = 0 },
        .{ .positions = &.{2047}, .expected = 2 },
        .{ .positions = &.{4094}, .expected = 0 },
        .{ .positions = &.{8192}, .expected = 0 },
        .{ .positions = &.{ 2047, 2047 }, .expected = 2 },
        .{ .positions = &.{ 2045, 65533 }, .expected = 0 },
        .{ .positions = &.{ 65533, 2045 }, .expected = 0 },
    }) |case| {
        _ = e.arena.reset(.retain_capacity);
        var plans = [_]Plan{
            .{ .stream = &first, .position = case.positions[0], .kind = .head, .held = 2 },
            .{ .stream = &second, .position = case.positions[case.positions.len - 1], .kind = .head, .held = 2 },
        };
        try allocate(&e, plans[0..case.positions.len]);
        for (plans[0..case.positions.len]) |p| try std.testing.expectEqual(case.expected, p.count());
    }
}

test "context-calibrated shared rounds preserve greedy and sampled token IDs" {
    const cfg_mod = @import("config.zig");
    const fake = @import("fake.zig");
    const Sampling = @import("sampling.zig").Sampling;
    const Run = struct {
        fn decode(profiles: []const cfg_mod.ContextCostProfile, drafts: bool, sampling: ?Sampling) ![2][]u32 {
            const gpa = std.testing.allocator;
            var cfg = try cfg_mod.Config.init(gpa, .{
                .exact_width = 6,
                .mtp = true,
                .speculate = true,
                .speculate_early = false,
                .drafts = 5,
                .draft_probabilities = true,
                .draft_streams = true,
                .hidden_rows = true,
                .max_streams = 2,
                .batch_rows = 16,
                .window_costs = &.{ .{ .width = 1, .ms = 10 }, .{ .width = 6, .ms = 15 } },
                .context_costs = profiles,
            }, 6, 5);
            defer cfg.deinit(gpa);
            var target: fake.Fake = .{ .gpa = gpa };
            defer target.deinit();
            var clock: fake.FixedClock = .{};
            var e = Engine.init(gpa, &cfg, target.backend(), clock.clock());
            defer e.deinit();
            var first = try Stream.init(gpa, .{ .id = "first", .prompt = &.{ 3, 1, 4, 1, 5 }, .max_new = 48, .eos = &.{}, .drafts = drafts, .sampling = sampling });
            defer first.deinit(gpa);
            var second = try Stream.init(gpa, .{ .id = "second", .prompt = &.{ 2, 7, 1, 8, 2, 8 }, .max_new = 48, .eos = &.{}, .drafts = drafts, .sampling = sampling });
            defer second.deinit(gpa);
            try e.addStream(&first);
            try e.addStream(&second);
            while (e.activeCount() > 0) try e.step();
            const first_tokens = try gpa.dupe(u32, first.emitted());
            errdefer gpa.free(first_tokens);
            return .{ first_tokens, try gpa.dupe(u32, second.emitted()) };
        }

        fn free(out: [2][]u32) void {
            for (out) |tokens_| std.testing.allocator.free(tokens_);
        }
    };
    const fast = &[_]f64{ 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 };
    const slow = &[_]f64{ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120 };
    const profiles = [_]cfg_mod.ContextCostProfile{
        .{ .max_context = 31, .streams = 1, .forward_ms = fast[0..6], .overhead_ms = 0 },
        .{ .max_context = 31, .streams = 2, .forward_ms = fast, .overhead_ms = 0 },
        .{ .min_context = 32, .max_context = 512, .streams = 1, .forward_ms = slow[0..6], .overhead_ms = 0 },
        .{ .min_context = 32, .max_context = 512, .streams = 2, .forward_ms = slow, .overhead_ms = 0 },
    };
    for ([_]?Sampling{ null, .{ .seed = 5, .temperature = 0.7, .top_p = 0.95 } }) |sampling| {
        const plain = try Run.decode(&.{}, false, sampling);
        defer Run.free(plain);
        const defaults = try Run.decode(&.{}, true, sampling);
        defer Run.free(defaults);
        const calibrated = try Run.decode(&profiles, true, sampling);
        defer Run.free(calibrated);
        for (plain, defaults, calibrated) |expected, unchanged, actual| {
            try std.testing.expectEqualSlices(u32, expected, unchanged);
            try std.testing.expectEqualSlices(u32, expected, actual);
        }
    }
}
