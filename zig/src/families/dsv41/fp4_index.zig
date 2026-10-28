//! Validate the opt-in packed-query indexer catalog as one atomic replacement of the existing launch ABI.
const std = @import("std");

const functions = [_][]const u8{ "_fp4qd_p2", "_index_score", "_index_score_rows", "_index_score_cand" };
const Variant = struct { function: []const u8, original_hash: []const u8, native_hash: []const u8 };
const Proof = struct {
    schema_version: u32,
    index_heads: usize,
    index_head_dim: usize,
    query_stride_bytes: usize,
    arch: u32,
    base_manifest_sha256: []const u8,
    variants: []const Variant,
};
const Entry = struct { @"fn": []const u8, hash: []const u8 };
const Catalog = struct { kernels: []const Entry };

fn replaces(name: []const u8) bool {
    for (functions) |function| if (std.mem.eql(u8, name, function)) return true;
    return false;
}

pub fn validate(gpa: std.mem.Allocator, old: []const u8, new: []const u8, proof: []const u8) !void {
    const options: std.json.ParseOptions = .{ .ignore_unknown_fields = true };
    const p = try std.json.parseFromSlice(Proof, gpa, proof, options);
    defer p.deinit();
    const v = p.value;
    if (v.schema_version != 2 or v.index_heads != 32 or v.index_head_dim != 128 or
        v.query_stride_bytes != 8192 or v.arch != 121) return error.BadFP4IndexerGeometry;
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(old, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    if (!std.mem.eql(u8, &hex, v.base_manifest_sha256)) return error.FP4IndexerBaselineMismatch;
    const before = try std.json.parseFromSlice(Catalog, gpa, old, options);
    defer before.deinit();
    const after = try std.json.parseFromSlice(Catalog, gpa, new, options);
    defer after.deinit();
    if (before.value.kernels.len != after.value.kernels.len) return error.IncompleteFP4IndexerCatalog;
    var changed: usize = 0;
    var present: [functions.len]bool = @splat(false);
    for (before.value.kernels) |entry| {
        var expected = entry.hash;
        if (replaces(entry.@"fn")) {
            const replacement = for (v.variants) |item| {
                if (std.mem.eql(u8, item.function, entry.@"fn") and
                    std.mem.eql(u8, item.original_hash, entry.hash)) break item;
            } else return error.IncompleteFP4IndexerCatalog;
            expected = replacement.native_hash;
            if (std.mem.eql(u8, expected, entry.hash)) return error.IncompleteFP4IndexerCatalog;
            changed += 1;
            for (functions, 0..) |name, i| if (std.mem.eql(u8, name, entry.@"fn")) {
                present[i] = true;
            };
        }
        var matches: usize = 0;
        for (after.value.kernels) |item| if (std.mem.eql(u8, item.@"fn", entry.@"fn") and
            std.mem.eql(u8, item.hash, expected))
        {
            matches += 1;
        };
        if (matches != 1) return error.IncompleteFP4IndexerCatalog;
    }
    if (changed != v.variants.len) return error.IncompleteFP4IndexerCatalog;
    for (present) |found| if (!found) return error.IncompleteFP4IndexerCatalog;
}

pub fn directory(gpa: std.mem.Allocator, io: std.Io, base: []const u8, requested: ?[]const u8) ![]const u8 {
    const path = requested orelse return base;
    if (path.len == 0) return error.BadFP4IndexerDirectory;
    const old_path = try std.fs.path.join(gpa, &.{ base, "aot.json" });
    defer gpa.free(old_path);
    const new_path = try std.fs.path.join(gpa, &.{ path, "aot.json" });
    defer gpa.free(new_path);
    const proof_path = try std.fs.path.join(gpa, &.{ path, "index-fp4.json" });
    defer gpa.free(proof_path);
    const old = try std.Io.Dir.cwd().readFileAlloc(io, old_path, gpa, .limited(1 << 24));
    defer gpa.free(old);
    const new = try std.Io.Dir.cwd().readFileAlloc(io, new_path, gpa, .limited(1 << 24));
    defer gpa.free(new);
    const proof = try std.Io.Dir.cwd().readFileAlloc(io, proof_path, gpa, .limited(1 << 20));
    defer gpa.free(proof);
    try validate(gpa, old, new, proof);
    return path;
}

test "packed queries require every scoring variant to be replaced together" {
    const a = std.testing.allocator;
    var entries: [5]Entry = undefined;
    var replacements: [4]Variant = undefined;
    var native: [5]Entry = undefined;
    for (functions, 0..) |name, i| {
        entries[i] = .{ .@"fn" = name, .hash = "old" };
        native[i] = .{ .@"fn" = name, .hash = "new" };
        replacements[i] = .{ .function = name, .original_hash = "old", .native_hash = "new" };
    }
    entries[4] = .{ .@"fn" = "untouched", .hash = "keep" };
    native[4] = entries[4];
    const old = try std.json.Stringify.valueAlloc(a, Catalog{ .kernels = &entries }, .{});
    defer a.free(old);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(old, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    var proof: Proof = .{ .schema_version = 2, .index_heads = 32, .index_head_dim = 128, .query_stride_bytes = 8192, .arch = 121, .base_manifest_sha256 = &hex, .variants = &replacements };
    const valid = try std.json.Stringify.valueAlloc(a, proof, .{});
    defer a.free(valid);
    const complete = try std.json.Stringify.valueAlloc(a, Catalog{ .kernels = &native }, .{});
    defer a.free(complete);
    try validate(a, old, complete, valid);
    native[2] = entries[2];
    const mixed = try std.json.Stringify.valueAlloc(a, Catalog{ .kernels = &native }, .{});
    defer a.free(mixed);
    try std.testing.expectError(error.IncompleteFP4IndexerCatalog, validate(a, old, mixed, valid));
    proof.query_stride_bytes = 2176;
    const wrong_stride = try std.json.Stringify.valueAlloc(a, proof, .{});
    defer a.free(wrong_stride);
    try std.testing.expectError(error.BadFP4IndexerGeometry, validate(a, old, complete, wrong_stride));
    proof.query_stride_bytes = 8192;
    proof.base_manifest_sha256 = "wrong";
    const wrong_base = try std.json.Stringify.valueAlloc(a, proof, .{});
    defer a.free(wrong_base);
    try std.testing.expectError(error.FP4IndexerBaselineMismatch, validate(a, old, complete, wrong_base));
}
