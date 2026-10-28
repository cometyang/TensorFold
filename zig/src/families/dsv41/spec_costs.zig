//! Explicit, versioned context cost profiles for an operator's measured deployment.
const std = @import("std");
const lanes = @import("lanes");

pub const Document = struct {
    schema_version: u32,
    family: []const u8,
    profiles: []const lanes.config.ContextCostProfile,
};

pub const Loaded = struct {
    parsed: std.json.Parsed(Document),

    pub fn parse(gpa: std.mem.Allocator, text: []const u8, rows: u32, streams: u32) !Loaded {
        const parsed = try std.json.parseFromSlice(Document, gpa, text, .{
            .ignore_unknown_fields = true,
            .allocate = .alloc_always,
        });
        errdefer parsed.deinit();
        const document = parsed.value;
        if (document.schema_version != 1 or !std.mem.eql(u8, document.family, "dsv41"))
            return error.BadSpecCostSchema;
        if (document.profiles.len == 0) return error.EmptySpecCosts;
        const Catalog = @TypeOf(@as(lanes.Config, undefined).context_costs);
        var checked = try Catalog.init(gpa, document.profiles, rows, streams);
        checked.deinit(gpa);
        return .{ .parsed = parsed };
    }

    pub fn read(gpa: std.mem.Allocator, io: std.Io, path: []const u8, rows: u32, streams: u32) !Loaded {
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
        defer gpa.free(text);
        return parse(gpa, text, rows, streams);
    }

    pub fn deinit(self: *Loaded) void {
        self.parsed.deinit();
        self.* = undefined;
    }
};

test "spec cost JSON owns its data and permits provenance" {
    const text = try std.testing.allocator.dupe(u8,
        \\{"schema_version":1,"family":"dsv41","profiles":[{"min_context":2048,
        \\"max_context":4096,"streams":1,"forward_ms":[10,11,12],"overhead_ms":2}],
        \\"provenance":{"clock_ceiling_mhz":2200}}
    );
    var loaded = try Loaded.parse(std.testing.allocator, text, 16, 2);
    std.testing.allocator.free(text);
    defer loaded.deinit();
    try std.testing.expectEqualStrings("dsv41", loaded.parsed.value.family);
    try std.testing.expectEqual(@as(f64, 12), loaded.parsed.value.profiles[0].forward_ms[2]);
}

test "spec cost JSON rejects incompatible, empty, and invalid profiles" {
    const gpa = std.testing.allocator;
    try std.testing.expectError(error.BadSpecCostSchema, Loaded.parse(gpa,
        \\{"schema_version":2,"family":"dsv41","profiles":[]}
    , 16, 2));
    try std.testing.expectError(error.BadSpecCostSchema, Loaded.parse(gpa,
        \\{"schema_version":1,"family":"other","profiles":[]}
    , 16, 2));
    try std.testing.expectError(error.EmptySpecCosts, Loaded.parse(gpa,
        \\{"schema_version":1,"family":"dsv41","profiles":[]}
    , 16, 2));
    try std.testing.expectError(error.BadContextCosts, Loaded.parse(gpa,
        \\{"schema_version":1,"family":"dsv41","profiles":[{"max_context":4096,
        \\"streams":1,"forward_ms":[0]}]}
    , 16, 2));
}
