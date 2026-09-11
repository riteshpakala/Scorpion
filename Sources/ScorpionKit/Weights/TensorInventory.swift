//
//  TensorInventory.swift
//  ScorpionKit
//
//  Header-only inventory of a resolved model: every tensor's name, dtype, shape and byte
//  range, grouped into analysis units (a single file, or a sharded set). Nothing here
//  knows about architectures — only file formats.
//

import Foundation

public struct TensorRecord: Sendable, Hashable {
    public let name: String
    public let file: RemoteFile
    public let rawDType: String
    public let shape: [Int]
    /// Absolute byte range within `file`.
    public let byteRange: Range<Int>

    public var dtype: TensorDType? { TensorDType(rawValue: rawDType) }
    public var elementCount: Int { shape.reduce(1, *) }
}

/// A set of files analyzed together: one checkpoint, adapter, or sharded set.
public struct WeightUnit: Sendable {
    public let name: String
    public let files: [RemoteFile]
    public let tensors: [String: TensorRecord]
    /// Merged `__metadata__` of all files in the unit.
    public let metadata: [String: String]
    public let headerBytes: Int

    public var byteSize: Int { files.compactMap(\.size).reduce(0, +) }
    public var tensorNames: [String] { tensors.keys.sorted { $0.naturalLess($1) } }
}

public struct FileInventoryNote: Codable, Sendable {
    public let path: String
    public let format: WeightFormat
    public let size: Int?
    public let tensorCount: Int?
    public let note: String?
}

public struct TensorInventory: Sendable {
    public let units: [WeightUnit]
    public let notes: [FileInventoryNote]
    public let ggufHeaders: [String: GGUFHeader]
    /// Risk / capability flags discovered from files alone (e.g. "pickle").
    public let flags: [String]
    public let headerBytes: Int
}

public struct InventoryReader: Sendable {
    let fetcher: RangeFetcher
    /// First request size; most safetensors headers fit, saving a round trip.
    var headerProbeBytes = 256 << 10

    public init(fetcher: RangeFetcher) { self.fetcher = fetcher }

    public func read(_ source: ResolvedSource) async throws -> TensorInventory {
        var notes: [FileInventoryNote] = []
        var flags: Set<String> = []
        var ggufHeaders: [String: GGUFHeader] = [:]
        var headerBytes = 0

        let weights = source.weightFiles
        if weights.isEmpty { throw SourceError.noWeightFiles(source.reference.original) }

        // Sharded sets: files listed in an index are grouped into one unit.
        var shardGroups: [String: [String]] = [:]  // index path → shard paths
        for index in source.files where index.path.hasSuffix(".safetensors.index.json") {
            if let data = try? await fetcher.fetchSmallFile(index.url, cacheable: source.pinned),
               let parsed = try? SafetensorsIndex.parse(data) {
                let dir = (index.path as NSString).deletingLastPathComponent
                shardGroups[index.path] = parsed.shards.map { dir.isEmpty ? $0 : dir + "/" + $0 }
            }
        }
        let shardedPaths = Set(shardGroups.values.flatMap { $0 })

        var headers: [String: SafetensorsHeader] = [:]
        for file in weights {
            switch file.format {
            case .safetensors:
                do {
                    let header = try await readSafetensorsHeader(file, cacheable: source.pinned)
                    headers[file.path] = header
                    headerBytes += header.dataStart
                    notes.append(FileInventoryNote(path: file.path, format: .safetensors, size: file.size,
                                                   tensorCount: header.entries.count, note: nil))
                } catch {
                    notes.append(FileInventoryNote(path: file.path, format: .safetensors, size: file.size,
                                                   tensorCount: nil, note: "header unreadable: \(error.localizedDescription)"))
                }
            case .gguf:
                if let (header, bytes) = try? await readGGUFHeader(file, cacheable: source.pinned) {
                    ggufHeaders[file.path] = header
                    headerBytes += bytes
                    notes.append(FileInventoryNote(path: file.path, format: .gguf, size: file.size,
                                                   tensorCount: header.tensors.count,
                                                   note: "GGUF: inventory only (quantized payloads not decoded in Phase 1)"))
                } else {
                    notes.append(FileInventoryNote(path: file.path, format: .gguf, size: file.size,
                                                   tensorCount: nil, note: "GGUF header unreadable"))
                }
            case .pickle:
                flags.insert("pickle")
                notes.append(FileInventoryNote(path: file.path, format: .pickle, size: file.size, tensorCount: nil,
                                               note: "pickle format — never deserialized (arbitrary code execution risk)"))
            case .onnx:
                notes.append(FileInventoryNote(path: file.path, format: .onnx, size: file.size, tensorCount: nil,
                                               note: "ONNX — not analyzed in Phase 1"))
            case .other:
                break
            }
        }

        var units: [WeightUnit] = []
        for (indexPath, shards) in shardGroups.sorted(by: { $0.key < $1.key }) {
            let files = shards.compactMap { p in weights.first { $0.path == p } }
            let hs = files.compactMap { headers[$0.path] }
            guard !hs.isEmpty else { continue }
            units.append(makeUnit(name: indexPath.replacingOccurrences(of: ".index.json", with: ""),
                                  files: files, headers: headers))
        }
        let singles = weights.filter { $0.format == .safetensors && !shardedPaths.contains($0.path) && headers[$0.path] != nil }
        for file in Self.dedupeVariants(singles) {
            units.append(makeUnit(name: file.path, files: [file], headers: headers))
        }
        return TensorInventory(units: units, notes: notes, ggufHeaders: ggufHeaders,
                               flags: flags.sorted(), headerBytes: headerBytes)
    }

    func makeUnit(name: String, files: [RemoteFile], headers: [String: SafetensorsHeader]) -> WeightUnit {
        var tensors: [String: TensorRecord] = [:]
        var metadata: [String: String] = [:]
        var bytes = 0
        for file in files {
            guard let h = headers[file.path] else { continue }
            bytes += h.dataStart
            metadata.merge(h.metadata) { a, _ in a }
            for e in h.entries {
                tensors[e.name] = TensorRecord(name: e.name, file: file, rawDType: e.rawDType,
                                               shape: e.shape, byteRange: h.absoluteRange(of: e))
            }
        }
        return WeightUnit(name: name, files: files, tensors: tensors, metadata: metadata, headerBytes: bytes)
    }

    /// Repos often ship the same weights in several precisions ("x.safetensors",
    /// "x.fp16.safetensors"). Keep one per stem — the smallest, since it's cheapest to fetch.
    static func dedupeVariants(_ files: [RemoteFile]) -> [RemoteFile] {
        let variantTokens = [".fp16", ".fp32", ".bf16", "-fp16", "_fp16", "-fp32", "_fp32", "-bf16", "_bf16", ".ema", "-ema"]
        func stem(_ path: String) -> String {
            var s = path.lowercased().replacingOccurrences(of: ".safetensors", with: "")
            for t in variantTokens { s = s.replacingOccurrences(of: t, with: "") }
            return s
        }
        var best: [String: RemoteFile] = [:]
        for f in files {
            let k = stem(f.path)
            if let cur = best[k], (cur.size ?? .max) <= (f.size ?? .max) { continue }
            best[k] = f
        }
        return best.values.sorted { $0.path < $1.path }
    }

    public func readSafetensorsHeader(_ file: RemoteFile, cacheable: Bool) async throws -> SafetensorsHeader {
        let probe = min(headerProbeBytes, file.size ?? headerProbeBytes)
        var prefix = try await fetcher.fetch(file.url, range: 0..<max(probe, 8), cacheable: cacheable)
        let n = try SafetensorsHeader.headerLength(fromPrefix: prefix)
        if prefix.count < 8 + n {
            prefix.append(try await fetcher.fetch(file.url, range: prefix.count..<(8 + n), cacheable: cacheable))
        }
        return try SafetensorsHeader.parse(prefix: prefix)
    }

    func readGGUFHeader(_ file: RemoteFile, cacheable: Bool) async throws -> (GGUFHeader, Int) {
        var want = 1 << 20
        let cap = 64 << 20
        while true {
            let size = min(want, file.size ?? want)
            let prefix = try await fetcher.fetch(file.url, range: 0..<size, cacheable: cacheable)
            do {
                return (try GGUFHeader.parse(prefix), prefix.count)
            } catch GGUFError.needMoreData(let n) {
                guard want < cap, size < (file.size ?? .max) else { throw GGUFError.needMoreData(n) }
                want = min(cap, max(want * 4, n + (1 << 20)))
            }
        }
    }
}
