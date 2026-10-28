//! Buffered CPU-wall speculative timings; recording never adds GPU synchronization or exposes token IDs.
const std = @import("std");

pub const max_streams = 16;
pub const max_drafts = 8;
const buffer_events = 4096;

pub const Window = struct {
    trace_id: u64 = 0,
    slot: usize = 0,
    context: usize = 0,
    proposed_context_end: usize = 0,
    rows: usize = 0,
    held_drafts: usize = 0,
    host_drafts: usize = 0,
    selected_drafts: usize = 0,
    drafts_enabled: bool = false,
    round_index: u64 = 0,
    first_round: bool = false,
    bucket_changed: bool = false,
    potential_accepted_drafts: usize = 0,
    confidence_count: usize = 0,
    confidence_logits: [max_drafts]f32 = @splat(0),
};

pub const Event = struct {
    event: enum { verify, draft, release },
    sequence: u64 = 0,
    wire_step: u64 = 0,
    max_context: usize = 0,
    max_context_end: usize = 0,
    bucket: usize = 0,
    streams: usize = 0,
    rows: usize = 0,
    steps: usize = 0,
    greedy: bool = true,
    graphs_enabled: bool = false,
    start_ns: u64 = 0,
    send_ns: u64 = 0,
    verify_ns: u64 = 0,
    readback_ns: u64 = 0,
    sample_ns: u64 = 0,
    pass_ns: u64 = 0,
    total_ns: u64 = 0,
    calibration_k: ?usize = null,
    windows: [max_streams]Window = @splat(.{}),
};

/// An accepted prefix before stop tokens and output budgets; it is not the scheduler's committed count.
pub fn potentialPrefix(drafts: []const u32, sampled: []const u32) usize {
    var n: usize = 0;
    for (drafts, sampled[0..drafts.len]) |d, s| {
        if (d != s) break;
        n += 1;
    }
    return n;
}

pub const Trace = struct {
    output: ?*Output = null,

    const Output = struct {
        gpa: std.mem.Allocator,
        io: std.Io,
        file: std.Io.File,
        writer: std.Io.File.Writer = undefined,
        bytes: [1 << 16]u8 = undefined,
        records: []Event,
        count: usize = 0,
        sequence: u64 = 0,
        next_request: u64 = 1,
        verify_events: u64 = 0,
        draft_events: u64 = 0,
        release_events: u64 = 0,
        failed: bool = false,
    };

    pub fn init(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !Trace {
        const records = try gpa.alloc(Event, buffer_events);
        errdefer gpa.free(records);
        const o = try gpa.create(Output);
        errdefer gpa.destroy(o);
        const file = try std.Io.Dir.cwd().createFile(io, path, .{});
        errdefer file.close(io);
        o.* = .{ .gpa = gpa, .io = io, .file = file, .records = records };
        o.writer = file.writer(io, &o.bytes);
        try o.writer.interface.writeAll(
            "{\"schema_version\":1,\"event\":\"header\",\"clock\":\"cpu_wall_ns\"," ++
                "\"gpu_sync_added\":false,\"acceptance\":\"potential_prefix_before_stop_and_budget\"," ++
                "\"buffer_events\":4096}\n",
        );
        try o.writer.interface.flush();
        return .{ .output = o };
    }

    pub fn enabled(self: *const Trace) bool {
        const o = self.output orelse return false;
        return !o.failed;
    }

    pub fn newRequest(self: *Trace) u64 {
        if (!self.enabled()) return 0;
        const o = self.output.?;
        const id = o.next_request;
        o.next_request += 1;
        return id;
    }

    pub fn record(self: *Trace, event: Event) void {
        if (!self.enabled()) return;
        const o = self.output.?;
        if (o.count == o.records.len) self.flush();
        if (o.failed) return;
        o.records[o.count] = event;
        o.records[o.count].sequence = o.sequence;
        o.count += 1;
        o.sequence += 1;
        switch (event.event) {
            .verify => o.verify_events += 1,
            .draft => o.draft_events += 1,
            .release => o.release_events += 1,
        }
    }

    /// Request release is the usual flush point; exceptionally long requests flush when the fixed buffer fills.
    pub fn flush(self: *Trace) void {
        if (!self.enabled()) return;
        const o = self.output.?;
        if (o.count == 0) return;
        flushOutput(o) catch |err| {
            o.failed = true;
            std.log.err("dsv41: speculative trace failed ({s}); trace is incomplete, inference continues", .{
                @errorName(err),
            });
        };
    }

    fn flushOutput(o: *Output) !void {
        for (o.records[0..o.count]) |*event| try writeEvent(&o.writer.interface, event);
        try o.writer.interface.flush();
        o.count = 0;
    }

    pub fn deinit(self: *Trace) void {
        const o = self.output orelse return;
        self.flush();
        if (!o.failed) writeSummary(o) catch |err| {
            std.log.err("dsv41: speculative trace summary failed ({s}); trace is incomplete", .{@errorName(err)});
        };
        o.file.close(o.io);
        const gpa = o.gpa;
        gpa.free(o.records);
        gpa.destroy(o);
        self.output = null;
    }

    fn writeSummary(o: *Output) !void {
        try o.writer.interface.print(
            "{{\"schema_version\":1,\"event\":\"summary\",\"complete\":true,\"events\":{d}," ++
                "\"verify_events\":{d},\"draft_events\":{d},\"release_events\":{d}}}\n",
            .{ o.sequence, o.verify_events, o.draft_events, o.release_events },
        );
        try o.writer.interface.flush();
    }
};

pub fn writeEvent(w: *std.Io.Writer, e: *const Event) !void {
    try w.print(
        "{{\"schema_version\":1,\"event\":\"{s}\",\"sequence\":{d},\"wire_step\":{d}," ++
            "\"max_context\":{d},\"max_context_end\":{d},\"bucket\":{d},\"streams\":{d},\"rows\":{d}," ++
            "\"steps\":{d},\"greedy\":{},\"graphs_enabled\":{},\"start_ns\":{d},\"send_ns\":{d},\"verify_ns\":{d}," ++
            "\"readback_ns\":{d},\"sample_ns\":{d},\"pass_ns\":{d},\"total_ns\":{d},\"calibration_k\":",
        .{
            @tagName(e.event), e.sequence,  e.wire_step,   e.max_context, e.max_context_end, e.bucket,
            e.streams,         e.rows,      e.steps,       e.greedy,      e.graphs_enabled,  e.start_ns,
            e.send_ns,         e.verify_ns, e.readback_ns, e.sample_ns,   e.pass_ns,         e.total_ns,
        },
    );
    if (e.calibration_k) |k| try w.print("{d}", .{k}) else try w.writeAll("null");
    try w.writeAll(",\"windows\":[");
    for (e.windows[0..e.streams], 0..) |*win, i| {
        if (i != 0) try w.writeByte(',');
        try writeWindow(w, win);
    }
    try w.writeAll("]}\n");
}

fn writeWindow(w: *std.Io.Writer, win: *const Window) !void {
    try w.print(
        "{{\"trace_id\":{d},\"slot\":{d},\"context\":{d},\"proposed_context_end\":{d},\"rows\":{d}," ++
            "\"held_drafts\":{d},\"host_drafts\":{d},\"selected_drafts\":{d},\"drafts_enabled\":{}," ++
            "\"round_index\":{d},\"first_round\":{},\"bucket_changed\":{},\"potential_accepted_drafts\":{d}," ++
            "\"confidence_logits\":[",
        .{
            win.trace_id,                  win.slot,            win.context,        win.proposed_context_end, win.rows,        win.held_drafts,
            win.host_drafts,               win.selected_drafts, win.drafts_enabled, win.round_index,          win.first_round, win.bucket_changed,
            win.potential_accepted_drafts,
        },
    );
    for (win.confidence_logits[0..win.confidence_count], 0..) |value, i| {
        if (i != 0) try w.writeByte(',');
        if (std.math.isFinite(value)) try w.print("{d}", .{value}) else try w.writeAll("null");
    }
    try w.writeAll("]}");
}

test "potential acceptance stops at the first mismatch and excludes the bonus row" {
    try std.testing.expectEqual(@as(usize, 2), potentialPrefix(&.{ 4, 5, 6 }, &.{ 4, 5, 7, 8 }));
    try std.testing.expectEqual(@as(usize, 3), potentialPrefix(&.{ 4, 5, 6 }, &.{ 4, 5, 6, 8 }));
    try std.testing.expectEqual(@as(usize, 0), potentialPrefix(&.{}, &.{8}));
    try std.testing.expectEqual(@as(usize, 0), potentialPrefix(&.{4}, &.{ 9, 8 }));
}

test "disabled trace records nothing and needs no allocator or IO" {
    var trace: Trace = .{};
    try std.testing.expect(!trace.enabled());
    try std.testing.expectEqual(@as(u64, 0), trace.newRequest());
    trace.record(.{ .event = .verify, .streams = max_streams + 1 });
    trace.flush();
    trace.deinit();
    try std.testing.expect(trace.output == null);
}

test "JSON event has bounded confidence data and explicit potential acceptance" {
    var e: Event = .{ .event = .verify, .streams = 1, .rows = 3, .bucket = 4096 };
    e.windows[0] = .{
        .trace_id = 3,
        .slot = 1,
        .context = 2048,
        .proposed_context_end = 2051,
        .rows = 3,
        .held_drafts = 2,
        .selected_drafts = 2,
        .drafts_enabled = true,
        .first_round = true,
        .potential_accepted_drafts = 1,
        .confidence_count = 2,
    };
    e.windows[0].confidence_logits[0] = 1.5;
    e.windows[0].confidence_logits[1] = std.math.nan(f32);
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try writeEvent(&out.writer, &e);
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, out.written(), .{});
    defer parsed.deinit();
    const obj = parsed.value.object;
    try std.testing.expectEqualStrings("verify", obj.get("event").?.string);
    try std.testing.expectEqual(@as(i64, 4096), obj.get("bucket").?.integer);
    const win = obj.get("windows").?.array.items[0].object;
    try std.testing.expect(win.get("drafts_enabled").?.bool);
    try std.testing.expectEqual(@as(i64, 1), win.get("potential_accepted_drafts").?.integer);
    const conf = win.get("confidence_logits").?.array.items;
    try std.testing.expectEqual(@as(usize, 2), conf.len);
    try std.testing.expectEqual(@as(f64, 1.5), conf[0].float);
    try std.testing.expect(conf[1] == .null);
    try std.testing.expect(win.get("tokens") == null);
}

test "buffered events sequence and counters advance without file IO" {
    var records: [3]Event = undefined;
    var output: Trace.Output = .{
        .gpa = undefined,
        .io = undefined,
        .file = undefined,
        .records = &records,
    };
    var trace: Trace = .{ .output = &output };
    try std.testing.expectEqual(@as(u64, 1), trace.newRequest());
    try std.testing.expectEqual(@as(u64, 2), trace.newRequest());
    trace.record(.{ .event = .verify });
    trace.record(.{ .event = .draft });
    trace.record(.{ .event = .release });
    try std.testing.expectEqual(@as(usize, 3), output.count);
    try std.testing.expectEqual(@as(u64, 2), records[2].sequence);
    try std.testing.expectEqual(@as(u64, 1), output.verify_events);
    try std.testing.expectEqual(@as(u64, 1), output.draft_events);
    try std.testing.expectEqual(@as(u64, 1), output.release_events);
    output.failed = true;
    trace.record(.{ .event = .verify });
    try std.testing.expectEqual(@as(usize, 3), output.count);
}

test "request flush preserves earlier records and clean close writes complete summary" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path, "trace.jsonl" });
    defer a.free(path);
    var trace = try Trace.init(a, io, path);
    defer trace.deinit();
    trace.record(.{ .event = .verify, .rows = 1, .calibration_k = 0 });
    trace.flush();
    trace.record(.{ .event = .draft, .rows = 5 });
    trace.record(.{ .event = .release });
    trace.flush();
    trace.deinit();
    const text = try tmp.dir.readFileAlloc(io, "trace.jsonl", a, .limited(1 << 20));
    defer a.free(text);
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    const names = [_][]const u8{ "header", "verify", "draft", "release", "summary" };
    for (names) |name| {
        const line = lines.next() orelse return error.MissingTraceRecord;
        const parsed = try std.json.parseFromSlice(std.json.Value, a, line, .{});
        defer parsed.deinit();
        const obj = parsed.value.object;
        try std.testing.expectEqualStrings(name, obj.get("event").?.string);
        if (std.mem.eql(u8, name, "summary")) {
            try std.testing.expect(obj.get("complete").?.bool);
            try std.testing.expectEqual(@as(i64, 3), obj.get("events").?.integer);
        }
    }
    try std.testing.expect(lines.next() == null);
}
